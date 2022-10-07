import os
import sys
import json
import torch
import pickle
import logging
import numpy as np
from IPython import embed
from pathlib import Path
from utils.zeshel_utils import N_ENTS_ZESHEL, N_MENTS_ZESHEL
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def _sort_by_score(indices, scores):
	"""
	Sort each row in scores array in decreasing order and also permute each row of ent_indices accordingly
	:param indices: 2-D numpy array of indices
	:param scores: 2-D numpy array of scores
	:return:
	"""
	assert indices.shape == scores.shape, f"ent_indices.shape ={indices.shape}  != ent_scores.shape = {scores.shape}"
	n,m = scores.shape
	scores = torch.tensor(scores)
	topk_scores, topk_idxs = torch.topk(scores, m)
	sorted_ent_indices = np.array([indices[i][topk_idxs[i]] for i in range(n)])
	
	return sorted_ent_indices, topk_scores


def combine_cur_crossenc_eval_results(dataset_name):
	try:
		
		topk = 500
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training"
		
		cur_file = f"{res_dir}/score_mats_eoe-0-last.ckpt/{dataset_name}/crossenc_w_cur_retrvr_nm={-1}_nm_start={0}_k_retvr=500_.txt"

		bienc_file = f"{res_dir}/eval/{dataset_name}/m=-1_k=500_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		comb_file = f"{res_dir}/eval/{dataset_name}/m=-1_k={topk}_CUR_500_negs_anc_nm_500_anc_ne_200_and_bienc_500_negs_comb_negs_eoe-0-last.ckpt.txt"
		
		# bienc_file = f"{res_dir}/eval/{dataset_name}/m=-1_k=100_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		# comb_file = f"{res_dir}/eval/{dataset_name}/m=-1_k={topk}_CUR_500_negs_anc_nm_500_anc_ne_200_and_bienc_100_negs_comb_negs_eoe-0-last.ckpt.txt"
		if os.path.exists(comb_file):
			return

		LOGGER.info("Reading CUR file")
		with open(cur_file, "r") as fin:
			cur_topk_preds = json.load(fin)
			
			LOGGER.info(f"Number of rows in indices : {len(cur_topk_preds['indices'])}")
			LOGGER.info(f"Number of rows in scores  : {len(cur_topk_preds['scores'])}")
		
		LOGGER.info("Reading bienc file")
		with open(bienc_file, "r") as fin:
			bienc_topk_preds = json.load(fin)
			
			LOGGER.info(f"Number of rows in indices : {len(bienc_topk_preds['indices'])}")
			LOGGER.info(f"Number of rows in scores  : {len(bienc_topk_preds['scores'])}")
	
		
		assert len(cur_topk_preds["indices"]) == N_MENTS_ZESHEL[dataset_name]
		assert len(cur_topk_preds["scores"]) == N_MENTS_ZESHEL[dataset_name]
		
		assert len(bienc_topk_preds["indices"]) == N_MENTS_ZESHEL[dataset_name]
		assert len(bienc_topk_preds["scores"]) == N_MENTS_ZESHEL[dataset_name]
		
		
		combined_topk_preds = {"indices":[], "scores":[]}
		
		n_ment = len(cur_topk_preds["indices"])
		
		for ment_idx in range(n_ment):
			curr_idx_to_score = {idx:score for idx, score in zip(cur_topk_preds["indices"][ment_idx], cur_topk_preds["scores"][ment_idx])}
			
			for idx,score in zip(bienc_topk_preds["indices"][ment_idx], bienc_topk_preds["scores"][ment_idx]):
				if idx in curr_idx_to_score:
					assert np.round(np.abs(score - curr_idx_to_score[idx]) ,4) == 0.0, f"ment_idx={ment_idx}, idx={idx}, score = {score} != curr_idx_to_score[idx] = {curr_idx_to_score[idx]}"
				else:
					curr_idx_to_score[idx] = score
			
			# curr_idx_to_score.update({idx:score for idx, score in zip(bienc_topk_preds["indices"][ment_idx], bienc_topk_preds["scores"][ment_idx])})
			
			
			comb_indices = list(curr_idx_to_score.keys())
			comb_scores = np.array([curr_idx_to_score[idx] for idx in comb_indices]).reshape(1, -1)
			comb_indices = np.array(comb_indices).reshape(1, -1)
			
			sorted_comb_indices, sorted_comb_scores = _sort_by_score(indices=comb_indices, scores=comb_scores)
			
			sorted_comb_indices = sorted_comb_indices[0][:topk].tolist()
			sorted_comb_scores = sorted_comb_scores[0][:topk].tolist()
			
			combined_topk_preds["indices"] += [sorted_comb_indices]
			combined_topk_preds["scores"] += [sorted_comb_scores]
			# embed()
			# input("")
		
		assert len(combined_topk_preds["indices"]) == N_MENTS_ZESHEL[dataset_name]
		assert len(combined_topk_preds["scores"]) == N_MENTS_ZESHEL[dataset_name]
		
	
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1 "))
			if not over_write: return
			
		LOGGER.info(f"Writing result to file : {comb_file}")
		
		with open(comb_file, "w") as fout:
			combined_topk_preds["other_args"] = {
				"cur_file": cur_file,
				"bienc_file": bienc_file,
				"topk": topk
			}
			json.dump(combined_topk_preds, fout)
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def main():
	from utils.zeshel_utils import get_zeshel_world_info
	dataset_name_vals = get_zeshel_world_info()
	
	# dataset_name_vals = [("train", "fallout")]
	for split_type, dataset_name in dataset_name_vals:
		# if dataset_name not in ["doctor_who", "starwars"]: continue
		if split_type in ["train", "valid"]:
			try:
				LOGGER.info(f"\n\nCombining files for {dataset_name}")
				combine_cur_crossenc_eval_results(dataset_name)
			except Exception as e:
				LOGGER.info(f"For data = {dataset_name}, error raised {e}")
	pass


if __name__ == "__main__":
	main()
