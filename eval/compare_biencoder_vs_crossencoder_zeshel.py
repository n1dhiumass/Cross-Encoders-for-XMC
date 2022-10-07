import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
import pickle


from eval.eval_utils import score_topk_preds, compute_label_embeddings, compute_overlap
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info

from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper

from utils.data_process import load_entities, load_mentions
from eval.run_gradient_based_search_w_cross_enc import GradientBasedInference


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)





def compare_bienc_vs_crossenc(biencoder, crossenc_ment_to_ent_scores, candidate_encoding,
							  mention_tokens_list, curr_gt_labels, top_k, grad_inf_obj):
	
	bienc_topk_preds  = []
	crossenc_topk_preds = []
	crossenc_topk_w_bienc_retr_preds = []
	crossenc_topk_w_grad_inf_preds = []
	candidate_encoding = candidate_encoding.t() # Take transpose for easier use later
	
	
	
	with torch.no_grad():
		biencoder.eval()
		
		for ment_idx, (ment_tokens, ment_gt_label) in tqdm(enumerate(zip(mention_tokens_list, curr_gt_labels)),
														   position=0, leave=True):
			ment_tokens = ment_tokens.unsqueeze(0)
			ment_tokens = ment_tokens.to(biencoder.device)
			ment_encoding = biencoder.encode_input(ment_tokens).cpu()
			bienc_scores = ment_encoding.mm(candidate_encoding)

			# bienc_pred = torch.argmax(bienc_scores, dim=1)

			crossenc_scores = crossenc_ment_to_ent_scores[ment_idx]
			# crossenc_pred = int(torch.argmax(crossenc_scores))
			
			crossenc_top_k_scores, crossenc_top_k_indices = crossenc_scores.topk(top_k)
		
			# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
			# If it were complete mention-entity matrix then we would have to use ment_idx
			bienc_top_k_scores, bienc_top_k_indices = bienc_scores.topk(top_k)

			# Re-rank top-k indices from bi-encoder model using cross-enc model
			temp = torch.zeros(crossenc_scores.shape) - 99999999999999
			temp[bienc_top_k_indices] = crossenc_scores[bienc_top_k_indices]
			# crossenc_w_bienc_retr_pred =  int(torch.argmax(temp))
			
			crossenc_w_bienc_retr_topk_scores, crossenc_w_bienc_retr_topk_indices = temp.topk(top_k)
			
	
			bienc_topk_preds += [(bienc_top_k_indices, bienc_top_k_scores)]
			crossenc_topk_preds += [(crossenc_top_k_indices.unsqueeze(0), crossenc_top_k_scores.unsqueeze(0))]
			crossenc_topk_w_bienc_retr_preds += [(crossenc_w_bienc_retr_topk_indices.unsqueeze(0), crossenc_w_bienc_retr_topk_scores.unsqueeze(0))]
			
	# Run gradient_based_inference outside of torch.no_grad
	assert isinstance(grad_inf_obj, GradientBasedInference)
	for ment_idx, (ment_tokens, ment_gt_label) in tqdm(enumerate(zip(mention_tokens_list, curr_gt_labels)),
														   position=0, leave=True):
		crossenc_grad_inf_indices, crossenc_grad_inf_scores, _all_res = grad_inf_obj.run(ment_idxs=[ment_idx],
																						 num_search_steps=top_k-1)
		crossenc_grad_inf_indices = crossenc_grad_inf_indices[0].unsqueeze(0)
		crossenc_grad_inf_scores = -1*crossenc_grad_inf_scores[0].unsqueeze(0)
		crossenc_topk_w_grad_inf_preds += [(crossenc_grad_inf_indices, crossenc_grad_inf_scores)]
	
	bienc_topk_preds = _get_indices_scores(bienc_topk_preds)
	crossenc_topk_preds = _get_indices_scores(crossenc_topk_preds)
	crossenc_topk_w_bienc_retr_preds = _get_indices_scores(crossenc_topk_w_bienc_retr_preds)
	crossenc_topk_w_grad_inf_preds = _get_indices_scores(crossenc_topk_w_grad_inf_preds)
	
	res = {"bienc": score_topk_preds(gt_labels=curr_gt_labels,
									 topk_preds=bienc_topk_preds),
		   "crossenc": score_topk_preds(gt_labels=curr_gt_labels,
										topk_preds=crossenc_topk_preds),
		   "crossenc_w_bienc_retrvr": score_topk_preds(gt_labels=curr_gt_labels,
													   topk_preds=crossenc_topk_w_bienc_retr_preds),
		   "crossenc_w_grad_inf": score_topk_preds(gt_labels=curr_gt_labels,
												   topk_preds=crossenc_topk_w_grad_inf_preds),
		   "bienc_vs_crossenc_overlap": compute_overlap(indices_list1=bienc_topk_preds["indices"],
														indices_list2=crossenc_topk_preds["indices"]),
		   "crossenc_grad_vs_crossenc_overlap": compute_overlap(indices_list1=crossenc_topk_w_grad_inf_preds["indices"],
																indices_list2=crossenc_topk_preds["indices"]),
		   }
	
	return res, (bienc_topk_preds, crossenc_topk_preds, crossenc_topk_w_bienc_retr_preds, crossenc_topk_w_grad_inf_preds)


def _get_indices_scores(topk_preds):
	"""
	Convert a list of indices,scores tuple to two list by concatenating all indices and all scores together.
	:param topk_preds: List of indices,scores tuple
	:return: dict with two keys "indices" and "scores" mapping to lists
	"""
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	return {"indices":indices, "scores":scores}


def run(biencoder, top_k, res_dir, data_info, bienc_config_path, misc, grad_inf_obj):
	try:
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fname = data_info
		
		LOGGER.info("Loading precomputed entity encodings computed using biencoder")
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		biencoder.eval()
		candidate_encoding = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=complete_entity_tokens_list,
													  batch_size=50)
		# candidate_encoding = np.load(data_fname["ent_embed_file"])
		
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			# test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			# entity_tokens_list = dump_dict["entity_tokens_list"]
			# (ment_to_ent_scores,
			#  test_data,
			#  mention_tokens_list,
			#  entity_id_list,
			#  entity_tokens_list) = pickle.load(fin)
			n_ment, n_ent = ment_to_ent_scores.shape
		
		# n_ment = 10
		mention_tokens_list = torch.LongTensor(mention_tokens_list)
		
		LOGGER.info("Loading all entities")
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		LOGGER.info("Loading test samples")
		test_data = load_mentions(mention_file=data_fname["ment_file"],
								  kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[:n_ment] if n_ment > 0 else test_data
		
		###########################################################################
		# This can be used for sanity check that gt labels and mentions are matched
		# First extract all mentions and tokenize them
		#
		# LOGGER.info(f"Tokenize {n_ment} test samples")
		# tokenizer = crossencoder.tokenizer
		# max_ent_length = 128
		# max_ment_length = 128
		# max_pair_length = 256
		#
		# mention_tokens_list = [get_context_representation(sample=mention,
		# 												 tokenizer=tokenizer,
		# 												 max_seq_length=max_ment_length)["ids"]
		# 						for mention in tqdm(test_data)]
		###########################################################################
		
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		local_id_to_ent_id = {idx:ent_id for idx, ent_id in enumerate(entity_id_list)}
	
		curr_gt_labels = [ent_id_to_local_id[mention["label_id"]] for mention in test_data]
		
		# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
		candidate_encoding = torch.Tensor(candidate_encoding[entity_id_list])

		res, (bi_preds, cross_preds, cross_w_bienc_retr_preds, cross_w_grad_inf_preds) \
			= compare_bienc_vs_crossenc(biencoder=biencoder,
										candidate_encoding=candidate_encoding,
										mention_tokens_list=mention_tokens_list,
										crossenc_ment_to_ent_scores=ment_to_ent_scores,
										top_k=top_k,
										curr_gt_labels=curr_gt_labels,
										grad_inf_obj=grad_inf_obj
									  )
		LOGGER.info("Done")
		Path(f"{res_dir}/bienc_vs_crossenc").mkdir(exist_ok=True, parents=True)
		
		with open(f"{res_dir}/bienc_vs_crossenc/preds_k={top_k}{misc}.json", "w") as fout:
			
			l1 = [ [(id2title[local_id_to_ent_id[idx]],score) for idx,score in zip(curr_indices, curr_scores)]
				   	for curr_indices, curr_scores in zip(bi_preds["indices"], bi_preds["scores"])]
			l2 = [ [(id2title[local_id_to_ent_id[idx]],score) for idx,score in zip(curr_indices, curr_scores)]
				   	for curr_indices, curr_scores in zip(cross_preds["indices"], cross_preds["scores"])]
			l3 = [ [(id2title[local_id_to_ent_id[idx]],score) for idx,score in zip(curr_indices, curr_scores)]
				   	for curr_indices, curr_scores in zip(cross_w_bienc_retr_preds["indices"], cross_w_bienc_retr_preds["scores"])]
			l4 = [ [(id2title[local_id_to_ent_id[idx]],score) for idx,score in zip(curr_indices, curr_scores)]
				   	for curr_indices, curr_scores in zip(cross_w_grad_inf_preds["indices"], cross_w_grad_inf_preds["scores"])]
			
			final_list = list(zip(test_data, l1,l2,l3, l4))
			
			preds = {"dataname": data_name, "n_ment": n_ment, "n_ent": n_ent, "top_k": top_k,
					 "bienc_path": bienc_config_path,
					 "crossenc_score_mat": data_fname["crossenc_ment_to_ent_scores"],
					 "comb_preds": final_list}
			
			json.dump(obj=preds, fp=fout, indent=4)
		
		with open(f"{res_dir}/bienc_vs_crossenc/res_k={top_k}{misc}.json", "w") as fout:
			res["dataname"] = data_name
			res["n_ment"] = n_ment
			res["n_ent"] = n_ent
			res["top_k"] = top_k
			res["bienc_path"] = bienc_config_path
			res["crossenc_score_mat"] = data_fname["crossenc_ment_to_ent_scores"]
			json.dump(obj=res, fp=fout)
		
		return res
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()


def main():
	# pretrained_dir = "../../BLINK_models"
	exp_id = "4_Zeshel"
	data_dir = "../../data/zeshel"
	# res_dir = f"../../results/{exp_id}"
	# Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	worlds = get_zeshel_world_info()
	parser = argparse.ArgumentParser( description='Compare exact biencoder model, exact cross-encoder model and '
												  'cross-encoder model with biencoder retriever. This requires '
												  'cross-encoder mention-entity scores to be pre-computed.')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions used in retrieval")

	parser.add_argument("--bi_model_config", type=str, required=True, help="Biencoder Model config file")
	parser.add_argument("--cross_model_config", type=str, required=True, help="Crossencoder Model config file")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	parser.add_argument("--lr", type=float, required=True, help="Learning rate for grad based search")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix for result file")
	
	args = parser.parse_args()
	data_name = args.data_name
	top_k = args.top_k
	
	res_dir = args.res_dir
	bi_model_config = args.bi_model_config
	cross_model_config = args.cross_model_config
	# misc = "_" + args.misc if args.misc != "" else ""
	lr = args.lr
	misc = "_" + args.misc if args.misc != "" else ""
	misc += f"_lr={lr}"
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
	
	# PARAMETERS = {
	# 		"biencoder_model": f"{pretrained_dir}/biencoder_wiki_large.bin",
	# 		"biencoder_config": f"{pretrained_dir}/biencoder_wiki_large.json",
	# 		"crossencoder_model": f"{pretrained_dir}/crossencoder_wiki_large.bin",
	# 		"crossencoder_config": f"{pretrained_dir}/crossencoder_wiki_large.json",
	# 	}
	# model_args = argparse.Namespace(**PARAMETERS)
	# (
	# 	biencoder,
	# 	biencoder_params,
	# 	crossencoder,
	# 	crossencoder_params
	# ) = load_models(model_args)
	
	with open(bi_model_config, "r") as fin:
		config = json.load(fin)
		biencoder = BiEncoderWrapper.load_model(config=config)
		
	with open(cross_model_config, "r") as fin:
		config = json.load(fin)
		crossencoder = CrossEncoderWrapper.load_model(config=config)
	
	iter_worlds = worlds[:4] if data_name == "all" else [("dummy", data_name)]
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Creating grad inf obj for  = {world_name}")
		
		with open(bi_model_config, "r") as fin:
			config = json.load(fin)
			biencoder_for_grad_search = BiEncoderWrapper.load_model(config=config)
			
		with open(cross_model_config, "r") as fin:
			config = json.load(fin)
			crossencoder_for_grad_search = CrossEncoderWrapper.load_model(config=config)
		
		grad_inf_obj = GradientBasedInference(crossencoder=crossencoder_for_grad_search,
											  biencoder=biencoder_for_grad_search,
											  dataset_name=data_name, data_fname=DATASETS[world_name],
											  quant_method="bienc", lr=lr)
		
		# grad_inf_obj.run(ment_idxs=[0], num_search_steps=top_k)
		
		
		LOGGER.info(f"Running inference for world = {world_name}")
		run(biencoder=biencoder, data_info=(world_name, DATASETS[world_name]),
			res_dir=f"{res_dir}/{world_name}", top_k=top_k, bienc_config_path=bi_model_config,
			misc=misc, grad_inf_obj=grad_inf_obj)
		#
		


if __name__ == "__main__":
	main()
