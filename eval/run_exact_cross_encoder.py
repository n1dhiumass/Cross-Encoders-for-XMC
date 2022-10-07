import os
import sys
import json
import wandb
import torch
import pickle
import logging
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from IPython import embed
from pathlib import Path

from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
from eval.eval_utils import score_topk_preds, compute_label_embeddings
from models.biencoder import BiEncoderWrapper

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

from eval.nsw_eval_zeshel import get_index, compute_ent_embeds_w_tfidf
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length


def avg_scores(list_of_score_dicts):
	try:
		metrics = {metric for score_dict in list_of_score_dicts for metric in score_dict}
		
		avg_scores = {}
		for metric in metrics:
			avg_scores[metric] = "{:.2f}".format(np.mean([float(score_dict[metric]) for score_dict in list_of_score_dicts]))
		
		return avg_scores
	except Exception as e:
		LOGGER.info("Exception raised in avg_scores")
		embed()
		raise e
	
	
def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	return {"indices":indices, "scores":scores}


def eval_perf_in_nbrhood(entity_scores, ent_embeds, max_nbrs, nsw_metric, gt_labels):
	
	try:
		if torch.is_tensor(entity_scores):
			entity_scores = entity_scores.cpu().numpy()
			
		n_ments, n_ents = entity_scores.shape
		num_hop_vals = [1,2,3,4,5,6,None]
		# num_hop_vals = [1,2]
		
		# 1. Build NSW graph on entities
		LOGGER.info(f"Building NSW graph over entities of shape {ent_embeds.shape} with max_nbrs = {max_nbrs}")
		index = get_index(
			index_path=None,
			embed_type="bienc", # Even if ent_embeds as using TF-IDF graph, passing this option ensures that we always use ent_embeds to build graph
			entity_file="",
			bienc_ent_embeds=ent_embeds,
			ment_to_ent_scores=None,
			max_nbrs=max_nbrs,
			graph_metric=nsw_metric
		)
		nsw_graph = index.get_nsw_graph_at_level(level=1)
		nx_graph = nx.from_dict_of_lists(nsw_graph)
		
		LOGGER.info(f"Running nbrhood eval for n_ments={n_ments}, n_ents={n_ents}")
		nbrhood_scores = {num_hops: [] for num_hops in num_hop_vals}
		for ment_id in range(n_ments):
			gt_node = gt_labels[ment_id]
			# 2. Retrieve up to k-hop nbrs from ground-truth
			tgt_path_lens = single_source_shortest_path_length(G=nx_graph, source=gt_node, cutoff=None)
			
			for num_hops in num_hop_vals:
				nbrs_upto_num_hops = [nbr for nbr, dist in tgt_path_lens.items() if num_hops is None or dist <= num_hops]
				
				nbr_scores = entity_scores[ment_id][nbrs_upto_num_hops]
				
				# Eval performance in this nbrhood by rank nbrs based on entity scores
				# nbrhood_scores[num_hops] += [(nbrs_upto_num_hops, nbr_scores)]
				curr_scores = score_topk_preds(
					gt_labels=[gt_node],
					topk_preds={
						"indices":[nbrs_upto_num_hops],
						"scores":[nbr_scores]
					}
				)
				curr_scores["num_nbrs"] = len(nbrs_upto_num_hops)
				nbrhood_scores[num_hops] += [curr_scores]
		
		# Average performance over all mentions
		res = {f"num_hops={num_hops}": avg_scores(nbrhood_scores[num_hops])
			   for num_hops in num_hop_vals}
		
		return res
	except Exception as e:
		LOGGER.info("Exception raised in eval_perf_in_nbrhood")
		embed()
		raise e
	
	
	


def run_exact_inf(dataset_name, data_fname, top_k, res_dir, misc):
	try:
		assert top_k > 1
		
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
		
		# FIXME: REMOVE THIS IF CONDITION
		if "rank_margin_neg" in data_fname["crossenc_ment_to_ent_scores"]:
			crossenc_ment_to_ent_scores = -1*crossenc_ment_to_ent_scores
			
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
	
		crossenc_top_k_scores, crossenc_top_k_indices = crossenc_ment_to_ent_scores.topk(top_k)
		crossenc_topk_preds = [(crossenc_top_k_indices, crossenc_top_k_scores)]
		crossenc_topk_preds = _get_indices_scores(crossenc_topk_preds)
		
		
		curr_res_dir = f"{res_dir}/{dataset_name}/exact_crossenc/m={n_ments}_k={top_k}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		json.dump(curr_gt_labels.tolist(), open(f"{curr_res_dir}/gt_labels.txt", "w"))
		json.dump(crossenc_topk_preds, open(f"{curr_res_dir}/crossenc_topk_preds.txt", "w"))
		
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			res = {"crossenc": score_topk_preds(gt_labels=curr_gt_labels,
												topk_preds=crossenc_topk_preds),
			}
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e





def run(dataset_name, data_fname, embed_type, bienc_config, res_dir, nsw_metric, misc):
	try:
		wandb.run.summary["status"] = 2
		if bienc_config is not None and os.path.isfile(bienc_config):
			with open(bienc_config, "r") as fin:
				config = json.load(fin)
				biencoder = BiEncoderWrapper.load_model(config=config)
			biencoder.eval()
		else:
			biencoder = None
		
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
		
		# FIXME: REMOVE THIS IF CONDITION
		if "rank_margin_neg" in data_fname["crossenc_ment_to_ent_scores"]:
			crossenc_ment_to_ent_scores = -1*crossenc_ment_to_ent_scores
			
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
	
		wandb.run.summary["status"] = 3
		########################################## COMPUTING ENTITY EMBEDDINGS #########################################
		if embed_type == "bienc":
			LOGGER.info("Computing entity encodings computed using biencoder")
			complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
			complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
			all_candidate_encoding = compute_label_embeddings(
				biencoder=biencoder,
				labels_tokens_list=complete_entity_tokens_list,
				batch_size=200
			)
			# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
			ent_embeds = all_candidate_encoding[entity_id_list].numpy()
		elif embed_type == "tfidf":
			ent_embeds = compute_ent_embeds_w_tfidf(entity_file=data_fname["ent_file"])
		else:
			raise Exception(f"Embed_type = {embed_type} not supported")
		
		################################################################################################################
		
		wandb.run.summary["status"] = 4
		max_nbr_vals = [5, 10, 20, 50, None]
		all_res = {}
		for max_nbr in tqdm(max_nbr_vals, total=len(max_nbr_vals)):
			curr_res = eval_perf_in_nbrhood(
				entity_scores=crossenc_ment_to_ent_scores,
				gt_labels=curr_gt_labels,
				ent_embeds=ent_embeds,
				max_nbrs=max_nbr,
				nsw_metric=nsw_metric
			)
			all_res[f"max_nbr={max_nbr}"] = curr_res
		
		curr_res_dir = f"{res_dir}/{dataset_name}/nbrhood_eval/m={n_ments}_embed={embed_type}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			json.dump(all_res, fout, indent=4)
			all_res["extra_info"] = {"bienc_config":bienc_config,
									 "res_dir":res_dir,
									 "data_fname":data_fname}
			LOGGER.info(json.dumps(all_res, indent=4))
		LOGGER.info("Done")
		wandb.run.summary["status"] = 5
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		
		
def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()

	parser = argparse.ArgumentParser( description='Use precomputed crossencoder score mats for exact inference')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	parser.add_argument("--embed_type", type=str, default="None", choices=["bienc", "tfidf", "None"], help="Method to embed entities")
	parser.add_argument("--top_k", type=int, default=100, help="top-k entities to recall wrt crossencoder scores")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	parser.add_argument("--bi_model_config", type=str, default="", help="Biencoder Model config file")
	parser.add_argument("--exact_only", type=int, choices=[0,1], default=1, help="1 to run exact inference only, 0 to run exact as well nbrhood inferences")
	parser.add_argument("--nsw_metric", type=str, default="l2", choices=["l2", "ip"], help="Metric/distance to use for building NSW")
	
	args = parser.parse_args()
	data_name = args.data_name
	res_dir = args.res_dir
	bienc_config = args.bi_model_config
	embed_type = args.embed_type
	nsw_metric = args.nsw_metric
	top_k = args.top_k
	misc = args.misc
	exact_only = bool(args.exact_only)
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
	
	iter_worlds = worlds if data_name == "all" else [("dummy", data_name)]
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		if exact_only:
			run_exact_inf(
				dataset_name=data_name,
				data_fname=DATASETS[world_name],
				top_k=top_k,
				res_dir=res_dir,
				misc=misc
			)
		else:
			wandb.init(project="NSW-Nbrhood-Eval",
			   dir="../../results/5_CrossEnc/PooledResults",
			   config={"goal": "Eval performance of crossencoder in nbrhood of gt entity in NSW graph",
					   "data_name":data_name,
					   "embed_type":embed_type,
					   "bi_model_config":bienc_config,
					   "matrix_dir":res_dir,
					   "CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]})
	
			wandb.run.summary["status"] = 1
			run(
				dataset_name=data_name,
				data_fname=DATASETS[world_name],
				embed_type=embed_type,
				bienc_config=bienc_config,
				res_dir=res_dir,
				misc=misc,
				nsw_metric=nsw_metric
			)
			wandb.run.summary["status"] = 6
	
	
if __name__ == "__main__":
	main()

