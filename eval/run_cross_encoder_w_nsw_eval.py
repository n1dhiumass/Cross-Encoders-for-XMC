import os
import sys
import json
import wandb
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from IPython import embed
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from eval.eval_utils import score_topk_preds
from eval.nsw_eval_zeshel import compute_ment_embeds, compute_ent_embeds, get_index, search_nsw_graph
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from utils.data_process import load_entities, load_mentions, get_context_representation, create_input_label_pair
from models.nearest_nbr import build_flat_or_ivff_index, HNSWWrapper

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _get_biencoder(bi_model_file):
	if bi_model_file == "":
		return None
	
	if bi_model_file.endswith(".json"):
		with open(bi_model_file, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
	else:
		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	
	return biencoder


def _get_crossencoder(cross_model_file):
	
	if cross_model_file.endswith(".json"):
		with open(cross_model_file, "r") as fin:
			config = json.load(fin)
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
	
	if isinstance(crossencoder, torch.nn.parallel.distributed.DistributedDataParallel):
		crossencoder = crossencoder.module.module
	assert isinstance(crossencoder, CrossEncoderWrapper), f"reranker is expected of type CrossEncoderWrapper but is of type = {type(crossencoder)}"
	
	return crossencoder


def run(
		cross_model_file,
		dataset_name,
		data_fname,
		n_ment_start,
		n_ment_arg,
		top_k,
		res_dir,
		misc,
		batch_size,
		bi_model_file,
		comp_budget,
		embed_type,
		beamsize,
		max_nbrs,
		graph_metric,
		graph_type,
		e2e_score_filename,
		arg_dict
):
	try:
		LOGGER.info(f"Starting Graph search with params \n {arg_dict}")
					
		entity_file = data_fname["ent_file"]
		mention_file = data_fname["ment_file"]
		entity_tokens_file = data_fname["ent_tokens_file"]
		assert top_k > 1
		
		biencoder = _get_biencoder(bi_model_file=bi_model_file)
		crossencoder = _get_crossencoder(cross_model_file=cross_model_file)
		if biencoder: biencoder.eval()
		crossencoder.eval()
		
		LOGGER.info(f"Bi encoder model device {biencoder.device if biencoder else None}")
		LOGGER.info(f"Cross encoder model device {crossencoder.device}")
		
		if crossencoder.device == torch.device("cpu"):
			wandb.alert(title="No GPUs found", text=f"{crossencoder.device}")
			raise Exception("No GPUs found!!!")
	
		
		######################### LOAD MENTIONS AND ENTITIES DATA ######################################################
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=entity_file)
		
		tokenizer = crossencoder.tokenizer
		
		mention_data = load_mentions(mention_file=mention_file, kb_id2local_id=kb_id2local_id)
		mention_data = mention_data[n_ment_start:n_ment_start+n_ment_arg] if n_ment_arg > 0 else mention_data
		
		# First extract all mentions and tokenize them
		tokenized_mentions = torch.LongTensor([get_context_representation(sample=mention,
																		  tokenizer=tokenizer,
																		  max_seq_length=MAX_MENT_LENGTH)["ids"]
											   for mention in tqdm(mention_data)])
		mentions_text = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
						 for ment_dict in mention_data]
		gt_labels = np.array([x["label_id"] for x in mention_data])
		tokenized_entities = torch.LongTensor(np.load(entity_tokens_file))
		
		n_ment, n_ents = len(mention_data), len(tokenized_entities)
		################################################################################################################
		
		######################### EMBED MENTIONS AND ENTITIES FOR BUILDING GRAPH #######################################
		ment_embeds = compute_ment_embeds(
			embed_type=embed_type,
			entity_file=entity_file,
			mentions=mentions_text,
			biencoder=biencoder,
			mention_tokens_list=tokenized_mentions
		)
		
		ent_embeds = compute_ent_embeds(
			embed_type=embed_type,
			biencoder=biencoder,
			entity_tokens_file=entity_tokens_file,
			entity_file=entity_file
		)
		nnbr_index = build_flat_or_ivff_index(embeds=ent_embeds, force_exact_search=True)
		_, init_ents = nnbr_index.search(ment_embeds, beamsize)
		

		################################################################################################################
		
		######################################### BUILD GRAPH ##########################################################
		
		LOGGER.info(f"Building a graph index over {n_ents} entities with embed shape {ent_embeds.shape} wiht max_nbrs={max_nbrs}")
		
		index = get_index(
			index_path=None,
			embed_type=embed_type,
			entity_file=entity_file,
			bienc_ent_embeds=ent_embeds,
			ment_to_ent_scores=None,
			max_nbrs=max_nbrs,
			graph_metric=graph_metric,
			graph_type=graph_type,
			e2e_score_filename=e2e_score_filename
		)
		
		LOGGER.info("Extracting lowest level graph from index")
		# Simulate graph search over this graph with pre-computed cross-encoder scores & Evaluate performance
		search_graph = index.get_nsw_graph_at_level(level=1)
		
		################################################################################################################
		
		######################### START GRAPH SEARCH USING CROSS-ENCODER MODEL ###########################################
		
		all_graph_topk_ents = []
		all_graph_topk_scores = []
		for ment_id in tqdm(range(n_ment), position=0, leave=True, total=n_ment):
			wandb.log({"ment_id": ment_id})
			wandb.log({"ment_id_frac": float(ment_id)/n_ment})
			def get_entity_scores(ent_ids):
				if len(ent_ids) == 0:
					return []
				all_pairs = [create_input_label_pair(input_token_idxs=tokenized_mentions[ment_id],
													 label_token_idxs=tokenized_entities[ent_id]).unsqueeze(0)
							 for ent_id in ent_ids]
				all_pairs = torch.cat(all_pairs).to(crossencoder.device)
			
				dataloader = DataLoader(TensorDataset(all_pairs), batch_size=batch_size, shuffle=False)
				all_scores_list = [crossencoder.score_candidate(batch_input, first_segment_end=MAX_MENT_LENGTH)
								   for (batch_input,) in dataloader]
				all_scores = torch.cat(all_scores_list)
				assert len(all_scores) == len(ent_ids), f"score shape = {all_scores.shape} does not match len(n_ment) = {len(ent_ids)}"
				
				return all_scores.detach().cpu().numpy()
			
			
			# Include gt_labels in init_ents for this mention if not already
			if gt_labels[ment_id] in set(init_ents[ment_id]):
				curr_init_ents = init_ents[ment_id]
			else:
				# Take first beamsize - 1 init_ents and replace last element of init_ents gt_labels[ment_id]
				curr_init_ents = init_ents[ment_id][:-1].tolist()
				curr_init_ents = curr_init_ents + [gt_labels[ment_id]]
		
			# TODO: Maybe also access scores for other entities that have been scored by cross--encoder
			# Find top entities using graph search
			graph_topk_scores , graph_topk_ents, graph_curr_num_score_comps = search_nsw_graph(
				nsw_graph=search_graph,
				entity_scores=get_entity_scores,
				approx_entity_scores_and_masked_nodes=(None,{}),
				topk=top_k,
				arg_beamsize=beamsize,
				init_ents=curr_init_ents,
				comp_budget=comp_budget,
				exit_at_local_minima_arg=False,
				pad_results=True
			)
		
			assert len(graph_topk_ents) > 0, f"No entity found in graph search for ment_id = {ment_id}, {entity_file}"
			while len(graph_topk_ents) < top_k:
				graph_topk_ents += graph_topk_ents
				graph_topk_scores += graph_topk_scores
			
			graph_topk_ents = graph_topk_ents[:top_k]
			graph_topk_scores = graph_topk_scores[:top_k]
			assert len(graph_topk_ents) == top_k
			all_graph_topk_ents += [graph_topk_ents]
			all_graph_topk_scores += [graph_topk_scores]
		
		LOGGER.info(f"Finished graph search for finding negs {entity_file}")
		
		
		crossenc_topk_preds_w_graph = [(all_graph_topk_ents, all_graph_topk_scores)]
		crossenc_topk_preds_w_graph = _get_indices_scores(crossenc_topk_preds_w_graph)
		crossenc_topk_preds_w_graph = {"indices":crossenc_topk_preds_w_graph["indices"].tolist(),
									 "scores":crossenc_topk_preds_w_graph["scores"].tolist()}
		
		
		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment_arg}_k={top_k}_g={graph_type}_e={embed_type}_{max_nbrs}_{beamsize}_{comp_budget}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		with open(f"{curr_res_dir}/gt_labels.txt", "w") as fout:
			json.dump(gt_labels.tolist(), fout)
		with open(f"{curr_res_dir}/crossenc_topk_preds_w_graph.txt", "w") as fout:
			json.dump(crossenc_topk_preds_w_graph, fout)
	
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			res = {
			   	"crossenc_w_graph": score_topk_preds(
					gt_labels=gt_labels,
					topk_preds=crossenc_topk_preds_w_graph
				),
			   	"extra_info": {
					"cross_model_file" : cross_model_file,
					"dataset_name" : dataset_name,
					"data_fname" : data_fname,
					"n_ment_arg" : n_ment_arg,
					"n_ment" : n_ment,
					"top_k" : top_k,
					"res_dir" : res_dir,
					"misc" : misc,
					"batch_size" : batch_size,
					"bi_model_file" : bi_model_file,
					"embed_type" : embed_type,
					"max_nbrs" : max_nbrs,
					"beamsize" : beamsize,
					"comp_budget" : comp_budget,
					"arg_dict":arg_dict
			   	}
			}
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e



def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	if torch.is_tensor(indices):
		indices, scores = torch.cat(indices), torch.cat(scores)
		indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	else:
		indices, scores = np.concatenate(indices), np.concatenate(scores)
		
	return {"indices":indices, "scores":scores}


def main():
	
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Search for top-scoring entity using cross-encoder using a graph')
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment_start", type=int, default=0, help="Start offset for mentions to use")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	
	parser.add_argument("--cross_model_file", type=str, required=True, help="Crossencoder Model config file or checkpoint file")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for cross-encoder calls")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	
	# Parameters for Graph search
	parser.add_argument("--embed_type", type=str, choices=["bienc", "tfidf"], required=True, help="Type of embeddings to use for building graph for search")
	parser.add_argument("--graph_metric", type=str, default="l2", choices=["l2", "ip"], help="Metric/distance to use for building NSW")
	parser.add_argument("--graph_type", type=str, default="nsw", choices=["knn", "nsw", "hnsw", "knn_e2e", "nsw_e2e", "rand"], help="Type of graph to use")
	parser.add_argument("--max_nbrs", type=int, required=True, help="max_mbrs parameter for building graph for search")
	parser.add_argument("--beamsize", type=int, required=True, help="Beamsize for search over graph")
	parser.add_argument("--top_k", type=int, default=64, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--comp_budget", type=int, default=250, help="Budget on number of crossencoder calls during graph search")
	parser.add_argument("--e2e_score_filename", type=str, default="", help="Pickle file containing entity-entity scores information")
	parser.add_argument("--bi_model_file", type=str, default="", help="Biencoder Model config file or checkpoint file")
	
	args = parser.parse_args()

	data_dir = args.data_dir
	data_name = args.data_name
	n_ment = args.n_ment
	n_ment_start = args.n_ment_start

	cross_model_file = args.cross_model_file
	batch_size = args.batch_size
	res_dir = args.res_dir
	misc = args.misc
	
	embed_type = args.embed_type
	max_nbrs = args.max_nbrs
	beamsize = args.beamsize
	graph_metric = args.graph_metric
	graph_type = args.graph_type
	e2e_score_filename = args.e2e_score_filename
	top_k = args.top_k
	comp_budget = args.comp_budget if args.comp_budget > 0 else None
	bi_model_file = args.bi_model_file
	

	Path(res_dir).mkdir(exist_ok=True, parents=True)
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	iter_worlds = worlds if data_name == "all" else [("dummy", data_name)]
	
	config = {
		"goal": "Run Graph Eval ",
		"cross_model_file" : cross_model_file,
		"dataset_name" : data_name,
		"data_fname" : DATASETS[data_name],
		"n_ment" : n_ment,
		"top_k" : top_k,
		"res_dir" : res_dir,
		"misc" : misc,
		"batch_size" : batch_size,
		"bi_model_file" : bi_model_file,
		"embed_type" : embed_type,
		"max_nbrs" : max_nbrs,
		"beamsize" : beamsize,
		"comp_budget" : comp_budget,
		"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
	}
	config.update(args.__dict__)
	
	try:
		wandb.init(
			project="Graph-Search-For-Eval",
			dir="../../results/5_CrossEnc/PooledResults",
			config=config
		)
	except:
		try:
			wandb.init(
				project="Graph-Search-For-Eval",
				dir="../../results/5_CrossEnc/PooledResults",
				config=config,
				settings=wandb.Settings(start_method="fork"),
			)
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(
				project="Graph-Search-For-Eval",
				dir="../../results/5_CrossEnc/PooledResults",
				config=config,
				mode="offline",
			)
	
	wandb.run.summary["status"] = 0
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		with torch.no_grad():
			run(
				cross_model_file=cross_model_file,
				dataset_name=world_name,
				data_fname=DATASETS[world_name],
				n_ment_start=n_ment_start,
				n_ment_arg=n_ment,
				top_k=top_k,
				res_dir=res_dir,
				misc=misc,
				batch_size=batch_size,
				bi_model_file=bi_model_file,
				comp_budget=comp_budget,
				embed_type=embed_type,
				beamsize=beamsize,
				max_nbrs=max_nbrs,
				graph_metric=graph_metric,
				graph_type=graph_type,
				e2e_score_filename=e2e_score_filename,
				arg_dict=args.__dict__
			)
	
	wandb.run.summary["status"] = 1
		


if __name__ == "__main__":
	main()

