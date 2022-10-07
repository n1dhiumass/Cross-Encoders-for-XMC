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


def combine_bi_plus_cross_eval_results():
	try:
		topk = 1000
		# dataset_name = "starwars"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,8001,2000)
		# ]
		# dir_list += [f"m=1824_k={topk}_1_eoe-0-last.ckpt_mstart_10000"]
		
		# dataset_name = "military"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,10001,2000)
		# ]
		# dir_list += [f"m=1063_k={topk}_1_eoe-0-last.ckpt_mstart_12000"]
		
		# dataset_name = "doctor_who"
		# dir_list = [
		# 	f"m=3000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,3001,3000)
		# ]
		# dir_list += [f"m=2334_k={topk}_1_eoe-0-last.ckpt_mstart_6000"]
		#
		# dataset_name = "american_football"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1,10)
		# ]
		# dir_list += [f"m=1898_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		# dataset_name = "final_fantasy"
		# dir_list = [
		# 	f"m=3000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		# ]
		# dir_list += [f"m=3041_k={topk}_1_eoe-0-last.ckpt_mstart_3000"]
		
		# dataset_name = "elder_scrolls"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		# ]
		# dir_list += [f"m=2275_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		dataset_name = "fallout"
		dir_list = [
			f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		]
		dir_list += [f"m=1286_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/eval/{dataset_name}"
		file_list = [f"{res_dir}/{curr_dir}/crossenc_topk_preds_w_bienc_retrvr.txt" for curr_dir in dir_list]

		
		combined_topk_preds = {"indices":[], "scores":[]}
		for curr_file in file_list:
			with open(curr_file, "r") as fin:
				curr_topk_preds = json.load(fin)
				
				combined_topk_preds["indices"] += curr_topk_preds["indices"]
				combined_topk_preds["scores"] += curr_topk_preds["scores"]
				LOGGER.info(f"curr_file {curr_file}")
				LOGGER.info(f"Number of rows in indices : {len(curr_topk_preds['indices'])}")
				LOGGER.info(f"Number of rows in scores  : {len(curr_topk_preds['scores'])}")
		
		LOGGER.info(f"Final number of rows in indices : {len(combined_topk_preds['indices'])}")
		LOGGER.info(f"Final number of rows in scores  : {len(combined_topk_preds['scores'])}")
		
		
		# temp_file = f"{res_dir}/m=-1_k=500_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		# with open(temp_file, "rb") as fin:
		# 	existing_res = json.load(fin)
		#
		#
		# for ment_idx, (exist_row, comb_row) in enumerate(zip(existing_res["indices"], combined_topk_preds["indices"])):
		# 	exist_row_set = set(exist_row[:150])
		# 	comb_row_set = set(comb_row[:150])
		# 	if len(set(exist_row_set) - set(comb_row_set)) != 0 or len(set(comb_row_set) - set(exist_row_set)) != 0:
		# 		LOGGER.info(f"{ment_idx} E-C: {set(exist_row_set) - set(comb_row_set)}")
		# 		LOGGER.info(f"{ment_idx} C-E: {set(comb_row_set) - set(exist_row_set)}\n")
		# 		for x in exist_row_set-comb_row_set:
		# 			_idx = existing_res["indices"][ment_idx].index(x)
		# 			score = existing_res["scores"][ment_idx][_idx]
		# 			LOGGER.info(f"E-C: Scores : {_idx}, {x} ->{score}")
		#
		# 		for x in comb_row_set-exist_row_set:
		# 			_idx = combined_topk_preds["indices"][ment_idx].index(x)
		# 			score = combined_topk_preds["scores"][ment_idx][_idx]
		# 			LOGGER.info(f"C-E: Scores : {_idx}, {x}->{score}")
		#
		# embed()
		
		comb_file = f"{res_dir}/m=-1_k={topk}_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1 "))
			if not over_write: return
			
		LOGGER.info(f"Writing result to file : {comb_file}")
		
		with open(comb_file, "w") as fout:
			json.dump(combined_topk_preds, fout)
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e

def combine_cur_crossenc_eval_results():
	try:
		dataset_name = "starwars"
		upper_ment_limit = 8000
		nment_per_run = 2000
		
		
		dataset_name = "military"
		upper_ment_limit = 10000
		nment_per_run = 2000
		
		dataset_name = "doctor_who"
		upper_ment_limit = 3000
		nment_per_run = 3000
		
		rel_file_list = [
			f"crossenc_w_cur_retrvr_nm={nment_per_run}_nm_start={mstart}_k_retvr=500_mstart_{mstart}.txt" for mstart in range(0,upper_ment_limit+1,nment_per_run)
		]
		final_m_start = upper_ment_limit + nment_per_run
		rel_file_list += [f"crossenc_w_cur_retrvr_nm={N_MENTS_ZESHEL[dataset_name] - final_m_start}_nm_start={final_m_start}_k_retvr=500_mstart_{final_m_start}.txt"]
		
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/score_mats_eoe-0-last.ckpt/{dataset_name}"
		file_list = [f"{res_dir}/{curr_rel_file}" for curr_rel_file in rel_file_list]

		
		combined_topk_preds = {"indices":[], "scores":[]}
		for curr_file in file_list:
			with open(curr_file, "r") as fin:
				curr_topk_preds = json.load(fin)
				
				combined_topk_preds["indices"] += curr_topk_preds["indices"]
				combined_topk_preds["scores"] += curr_topk_preds["scores"]
				LOGGER.info(f"curr_file {curr_file}")
				LOGGER.info(f"Number of rows in indices : {len(curr_topk_preds['indices'])}")
				LOGGER.info(f"Number of rows in scores  : {len(curr_topk_preds['scores'])}")
		
		LOGGER.info(f"Final number of rows in indices : {len(combined_topk_preds['indices'])}")
		LOGGER.info(f"Final number of rows in scores  : {len(combined_topk_preds['scores'])}")
		
		assert len(combined_topk_preds["indices"]) == N_MENTS_ZESHEL[dataset_name]
		assert len(combined_topk_preds["scores"]) == N_MENTS_ZESHEL[dataset_name]
		comb_file = f"{res_dir}/crossenc_w_cur_retrvr_nm={-1}_nm_start={0}_k_retvr=500_.txt"
	
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1 "))
			if not over_write: return
			
		LOGGER.info(f"Writing result to file : {comb_file}")
		
		with open(comb_file, "w") as fout:
			json.dump(combined_topk_preds, fout)
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e

def combine_nsw_eval_results():
	try:
		dataset_name = "military"
		dir_list = [
			f"m=1000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,11001,1000)
		]
		dir_list += ["m=1063_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_12K"]
		
		# dataset_name = "starwars"
		# dir_list = [
		# 	f"m=1000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,10001,1000)
		# ]
		# dir_list += ["m=824_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_11K"]
		#
		# dataset_name = "doctor_who"
		# dir_list = [
		# 	f"m=2000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,4001,2000)
		# ]
		# dir_list += ["m=2334_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_6000"]
		
		# dataset_name = "final_fantasy"
		# dir_list = [
		# 	f"m=2000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,2001,2000)
		# ]
		# dir_list += ["m=2041_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_4000"]
		
		# dataset_name = "american_football"
		# dir_list = [
		# 	f"m=1000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,2001,1000)
		# ]
		# dir_list += ["m=898_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_3000"]
		
		
		# dataset_name = "elder_scrolls"
		# dir_list = [
		# 	f"m=1000_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_{x}" for x in range(0,3001,1000)
		# ]
		# dir_list += ["m=275_k=100_g=knn_e2e_e=bienc_20_5_1000_mstart_4000"]
		
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/eval/{dataset_name}"
		file_list = [f"{res_dir}/{curr_dir}/crossenc_topk_preds_w_graph.txt" for curr_dir in dir_list]

		
		combined_topk_preds = {"indices":[], "scores":[]}
		for curr_file in file_list:
			with open(curr_file, "r") as fin:
				curr_topk_preds = json.load(fin)
				
				combined_topk_preds["indices"] += curr_topk_preds["indices"]
				combined_topk_preds["scores"] += curr_topk_preds["scores"]
				LOGGER.info(f"Number of rows in indices : {len(curr_topk_preds['indices'])}")
				LOGGER.info(f"Number of rows in scores  : {len(curr_topk_preds['scores'])}")
		
		LOGGER.info(f"Final number of rows in indices : {len(combined_topk_preds['indices'])}")
		LOGGER.info(f"Final number of rows in scores  : {len(combined_topk_preds['scores'])}")
		
		assert len(combined_topk_preds["scores"]) == len(combined_topk_preds["indices"])
		assert len(combined_topk_preds["scores"]) == N_MENTS_ZESHEL[dataset_name]
		
		comb_file = f"{res_dir}/m=-1_k=100_g=knn_e2e_e=bienc_20_5_1000_/crossenc_topk_preds_w_graph.txt"
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1"))
			if not over_write: return
			
		LOGGER.info(f"Writing result to file : {comb_file}")
		
		
		with open(comb_file, "w") as fout:
			json.dump(combined_topk_preds, fout)
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def combine_e2e_eval_results():
	try:
		dataset_name = "doctor_who"
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/" \
				  f"m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt/{dataset_name}"
		
		topk = 1000
		
		
		# Variables that will take different values across files
		comb_e2e_scores_list = []
		comb_topk_ents = []
		comb_n_ent_x = 0
		comb_entity_id_list_x = []
		comb_arg_dict = []
		
		# Variables that should take fixed values across files
		comb_n_ent_y = N_ENTS_ZESHEL[dataset_name]
		comb_token_opt = "m2e"
		comb_entity_tokens_list = None
		comb_entity_id_list_y = np.arange(N_ENTS_ZESHEL[dataset_name])
		
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{10}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,11,10)]
		
		# For YuGiOh
		# emb_type = "none"
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{2000}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,8001,2000)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{31}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{10000}.pkl"]
		
		# # For military
		# assert dataset_name  == "military"
		# emb_type = "none"
		# step = 2000
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{step}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,102001,step)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{520}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{104000}.pkl"]
		
		# # For star_trek
		# assert dataset_name  == "star_trek"
		# emb_type = "none"
		# step = 2000
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{step}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,32001,step)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{430}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{34000}.pkl"]
		
		# # For pro_wrestling
		# assert dataset_name  == "pro_wrestling"
		# emb_type = "none"
		# step = 2000
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{step}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,8001,step)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{133}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{10000}.pkl"]
		
		# For pro_wrestling
		assert dataset_name  == "doctor_who"
		emb_type = "none"
		step = 2000
		file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{step}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{xstart}.pkl"
					 for xstart in range(0,38001,step)]
		file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{281}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_x_start_{40000}.pkl"]
		
		
		# # For military
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{20000}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,60001,20000)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{24520}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_{80000}.pkl"]
		
		# # For starwars
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{20000}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_{xstart}.pkl"
		# 			 for xstart in range(0,40001,20000)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{27056}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_{60000}.pkl"]
		
		# comb_file = f"{res_dir}/ent_to_ent_scores_n_e_{comb_n_ent_x}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_.pkl"
		
		# # For pro_wrestling
		# emb_type="bienc"
		# "ent_to_ent_scores_n_e_400x10133_topk_1000_embed_tfidf_m2e__tfidf_cluster_alt_xstart_800.pkl"
		# file_list = [f"{res_dir}/ent_to_ent_scores_n_e_{1000}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_kmed_cluster_alt_xstart_{xstart}.pkl"
		# 			 for xstart in range(0,9001,1000)]
		# file_list += [f"{res_dir}/ent_to_ent_scores_n_e_{133}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_kmed_cluster_alt_xstart_{10000}.pkl"]
		
		
		for curr_file in file_list:
			with open(curr_file, "rb") as fin:
				res = pickle.load(fin)
				
				ent_to_ent_scores = res["ent_to_ent_scores"]
				topk_ents = res["topk_ents"]
				n_ent_x = res["n_ent_x"]
				n_ent_y = res["n_ent_y"]
				token_opt = res["token_opt"]
				entity_id_list_x = res["entity_id_list_x"]
				entity_id_list_y = res["entity_id_list_y"]
				entity_tokens_list = res["entity_tokens_list"]
				arg_dict = res["arg_dict"]
				
				comb_e2e_scores_list += [ent_to_ent_scores]
				comb_topk_ents += [topk_ents]
				comb_n_ent_x += n_ent_x
				comb_entity_id_list_x += [entity_id_list_x]
				
				comb_arg_dict += [arg_dict]
				
				# These values should be same across all files
				assert comb_token_opt == token_opt
				assert comb_entity_tokens_list is None or (comb_entity_tokens_list == entity_tokens_list).all()
				assert comb_n_ent_y == n_ent_y
				assert entity_tokens_list.shape[0] == N_ENTS_ZESHEL[dataset_name]
				assert (comb_entity_id_list_y == entity_id_list_y).all()
				comb_n_ent_y = n_ent_y
				comb_token_opt = token_opt
				comb_entity_tokens_list = entity_tokens_list
				
				LOGGER.info(f"Shape of current e2e matrix : {ent_to_ent_scores.shape}")
		
		comb_e2e_scores = torch.cat(comb_e2e_scores_list)
		comb_topk_ents = np.concatenate(comb_topk_ents)
		comb_entity_id_list_x = np.concatenate(comb_entity_id_list_x)
		
		# TODO: Assert that other args match across all arg_dicts
		comb_arg_dict = comb_arg_dict[-1]
		comb_arg_dict["n_ent_x_start"] = 0
		comb_arg_dict["n_ent_x"] = comb_n_ent_x
		comb_arg_dict["misc"] = ""
		
	
		assert comb_e2e_scores.shape == (comb_n_ent_x, topk)
		assert comb_topk_ents.shape == (comb_n_ent_x, topk)
		assert comb_entity_id_list_x.shape == (comb_n_ent_x,)
		

		LOGGER.info(f"Final shape of e2e matrix: {comb_e2e_scores.shape}")
		comb_file = f"{res_dir}/ent_to_ent_scores_n_e_{comb_n_ent_x}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_.pkl"
		# comb_file = f"{res_dir}/ent_to_ent_scores_n_e_{comb_n_ent_x}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_{emb_type}_{comb_token_opt}_kmed_cluster_alt.pkl" # FIXME : This is only for anchor2entity score computation
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1"))
			if not over_write: return

		LOGGER.info(f"Writing result to file : {comb_file}")

		with open(comb_file, "wb") as fout:
			comb_res = {
				"ent_to_ent_scores":comb_e2e_scores,
				"ent_to_ent_scores.shape":comb_e2e_scores.shape,
				"topk_ents":comb_topk_ents,
				"n_ent_x": comb_n_ent_x,
				"n_ent_y": comb_n_ent_y,
				"token_opt": comb_token_opt,
				"entity_id_list_x":comb_entity_id_list_x,
				"entity_id_list_y":comb_entity_id_list_y,
				"entity_tokens_list":comb_entity_tokens_list,
				"arg_dict":comb_arg_dict
			}
			pickle.dump(comb_res, fout)
	
			
		
		# # Debug
		# # Compare with entire result computed without chunking
		# debug_comb_file = f"{res_dir}/ent_to_ent_scores_n_e_{comb_n_ent_x}x{N_ENTS_ZESHEL[dataset_name]}_topk_{topk}_embed_bienc_{comb_token_opt}_x_start_0.pkl"
		#
		#
		# # PICK UP HERE
		# #
		# # - Perhaps the bug is in how entity pairs are constructed and tokenized and maybe that part of code does not take n_ent_x_start offset correctly
		#
		#
		# with open(debug_comb_file, "rb") as fin:
		# 	res = pickle.load(fin)
		#
		# 	ent_to_ent_scores = res["ent_to_ent_scores"]
		# 	topk_ents = res["topk_ents"]
		# 	n_ent_x = res["n_ent_x"]
		# 	n_ent_y = res["n_ent_y"]
		# 	token_opt = res["token_opt"]
		# 	entity_id_list_x = res["entity_id_list_x"]
		# 	entity_id_list_y = res["entity_id_list_y"]
		# 	entity_tokens_list = res["entity_tokens_list"]
		# 	arg_dict = res["arg_dict"]
		#
		#
		#
		# for ent_idx, (exist_row, comb_row) in enumerate(zip(topk_ents, comb_topk_ents)):
		# 	exist_row_set = set(exist_row)
		# 	comb_row_set = set(comb_row)
		# 	if len(set(exist_row_set) - set(comb_row_set)) != 0 or len(set(comb_row_set) - set(exist_row_set)) != 0:
		# 		LOGGER.info(f"{ent_idx} E-C: {set(exist_row_set) - set(comb_row_set)}")
		# 		LOGGER.info(f"{ent_idx} C-E: {set(comb_row_set) - set(exist_row_set)}\n")
		# 		for x in exist_row_set-comb_row_set:
		# 			_idx = topk_ents[ent_idx].index(x)
		# 			score = topk_ents[ent_idx][_idx]
		# 			LOGGER.info(f"E-C: Scores : {_idx}, {x} ->{score}")
		#
		# 		for x in comb_row_set-exist_row_set:
		# 			_idx = comb_topk_ents[ent_idx].index(x)
		# 			score = comb_topk_ents[ent_idx][_idx]
		# 			LOGGER.info(f"C-E: Scores : {_idx}, {x}->{score}")
		#
		# # assert ent_to_ent_scores == comb_e2e_scores
		# ent_to_ent_scores = ent_to_ent_scores.cpu().numpy().tolist()
		# comb_e2e_scores = comb_e2e_scores.cpu().numpy().tolist()
		#
		# for ent_idx, (exist_row, comb_row) in enumerate(zip(ent_to_ent_scores, comb_e2e_scores)):
		# 	exist_row = np.round(exist_row, 2).tolist()
		# 	comb_row = np.round(comb_row, 2).tolist()
		#
		# 	exist_row_set = set(exist_row)
		# 	comb_row_set = set(comb_row)
		# 	if len(set(exist_row_set) - set(comb_row_set)) != 0 or len(set(comb_row_set) - set(exist_row_set)) != 0:
		# 		LOGGER.info(f"{ent_idx} E-C: {sorted(set(exist_row_set) - set(comb_row_set))}")
		# 		LOGGER.info(f"{ent_idx} C-E: {sorted(set(comb_row_set) - set(exist_row_set))}\n")
		# 		for x in exist_row_set-comb_row_set:
		# 			_idx = exist_row.index(x)
		# 			score = exist_row[_idx]
		# 			LOGGER.info(f"E-C: Scores : {_idx}, {x} ->{score}")
		#
		# 		for x in comb_row_set-exist_row_set:
		# 			_idx = comb_row.index(x)
		# 			score = comb_row[_idx]
		# 			LOGGER.info(f"C-E: Scores : {_idx}, {x}->{score}")
		#
		#
		# 	# LOGGER.info("Intentional embed")
		# 	# embed()
		#
		# assert n_ent_x == comb_n_ent_x
		# assert n_ent_y == comb_n_ent_y
		# assert token_opt == comb_token_opt
		# assert (entity_id_list_x == comb_entity_id_list_x).all()
		# assert (entity_id_list_y == comb_entity_id_list_y).all()
		# assert (entity_tokens_list == comb_entity_tokens_list).all()
		# # assert arg_dict == comb_arg_dict
		
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def combine_m2e_eval_results():
	try:
		# "joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/score_mats_model-3-24599.0--75.46.ckpt"
		#
		# "joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-19679.0--77.23.ckpt"
		#
		# "m=cross_enc_l=ce_neg=bienc_distill_s=1234_crossenc_w_embeds/model/score_mats_model-1-12279.0-1.91.ckpt"
		#
		
		# res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/score_mats_model-3-24599.0--75.46.ckpt"
		# res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt"
		# res_dir = f"../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_cls_w_lin_tfidf_hard_negs/score_mats_model-1-12279.0--90.95.ckpt"
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/score_mats_eoe-0-last.ckpt"
		
		# military starwars doctor_who american_football world_of_warcraft fallout final_fantasy pro_wrestling ice_hockey muppets elder_scrolls coronation_street
		dataset_name = "world_of_warcraft"
		step = 50
		file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
					 for mstart in range(0, 500, step)]
		
		# dataset_name = "yugioh"
		# step = 100
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(0, 3201, step)]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{74}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{3350}.pkl"]
		#
		# dataset_name = "pro_wrestling"
		# step = 250
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(0, 1001, step)]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{142}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{1250}.pkl"]
		
		# dataset_name = "military"
		#
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{1000}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"]
		#
		# step = 50
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1000, 1201, step)]
		#
		# step = 1
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1250, 1300, step)]
		#
		# step = 50
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1300, 1901, step)]
		# step = 10
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1950, 2491, step)]
		#
		# dataset_name = "star_trek"
		# step = 50
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{2500}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(2500, 4151, step)]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{27}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{4200}.pkl"]
		
		comb_ment_to_ent_scores_list = []
		comb_test_data_list = []
		comb_mention_tokens_list = []
		
		# Values that should remain fixed across all files
		comb_entity_id_list = np.arange(N_ENTS_ZESHEL[dataset_name])
		comb_entity_tokens_list = None
		comb_arg_dict = []
		
		for curr_file in file_list:
			with open(curr_file, "rb") as fin:
				dump_dict = pickle.load(fin)
				ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				test_data = dump_dict["test_data"]
				mention_tokens_list = dump_dict["mention_tokens_list"]
				entity_id_list = dump_dict["entity_id_list"]
				entity_tokens_list = dump_dict["entity_tokens_list"]
				arg_dict = dump_dict["arg_dict"]
			
			comb_ment_to_ent_scores_list += [ment_to_ent_scores]
			comb_test_data_list += [test_data]
			comb_mention_tokens_list += [mention_tokens_list]
			
			comb_arg_dict += [arg_dict]
			
			assert comb_entity_tokens_list is None or (comb_entity_tokens_list == entity_tokens_list).all()
			assert (comb_entity_id_list == entity_id_list).all()
			
			LOGGER.info(f"Shape of current m2e matrix : {ment_to_ent_scores.shape}")
			
		comb_ment_to_ent_scores = torch.cat(comb_ment_to_ent_scores_list)
		comb_test_data = [x for xs in comb_test_data_list for x in xs] # Concat lists present in the list
		comb_mention_tokens = [x for xs in comb_mention_tokens_list for x in xs]
		
		LOGGER.info(f"Shape of final m2e matrix : {comb_ment_to_ent_scores.shape}")
		total_n_ments = comb_ment_to_ent_scores.shape[0]
		assert total_n_ments == len(comb_test_data), f"total_n_ments = {total_n_ments} != len(comb_test_data) = {len(comb_test_data)}"
		assert total_n_ments == len(comb_mention_tokens), f"total_n_ments = {total_n_ments} != len(comb_test_data) = {len(comb_test_data)}"
		
		comb_file = f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{total_n_ments}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"
		
		
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1\n"))
			if not over_write: return

		LOGGER.info(f"Writing result to file : {comb_file}")
		with open(comb_file, "wb") as fout:
			
			res = {
				"ment_to_ent_scores":comb_ment_to_ent_scores,
				"ment_to_ent_scores.shape":comb_ment_to_ent_scores.shape,
				"test_data":comb_test_data,
				"mention_tokens_list":comb_mention_tokens,
				"entity_id_list":comb_entity_id_list,
				"entity_tokens_list":comb_entity_tokens_list,
				"arg_dict":comb_arg_dict[-1]
			}
			pickle.dump(res, fout)
			
		
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e



def main():
	# combine_bi_plus_cross_eval_results()
	# combine_nsw_eval_results()
	# combine_e2e_eval_results()
	# combine_m2e_eval_results()
	combine_cur_crossenc_eval_results()
	pass


if __name__ == "__main__":
	main()
