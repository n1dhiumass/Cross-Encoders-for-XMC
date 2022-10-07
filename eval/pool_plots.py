import os
import sys
import pickle
import logging
import itertools
import numpy as np
import torch
from IPython import embed
from tqdm import tqdm
from pathlib import Path
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
from eval.visualize_matrix_zeshel import visualize_hist
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

DATA_DIR = "../../data/zeshel"
BASE_RES_DIR = "../../results"
WORLDS = get_zeshel_world_info()


def move_files(all_out_files, pooled_res_dir):
	
	Path(pooled_res_dir).mkdir(exist_ok=True, parents=True)
	
	for i, curr_train_setting in enumerate(all_out_files):
		abs_filename = all_out_files[curr_train_setting]
		rel_filename = abs_filename.split("/")[-1]
		if os.path.isfile(abs_filename):
			command = f"cp {abs_filename} {pooled_res_dir}/"
			os.system(command)
			
			command = f"rm {pooled_res_dir}/{curr_train_setting}.pdf "
			command = f"mv {pooled_res_dir}/{rel_filename} {pooled_res_dir}/{curr_train_setting}.pdf "
			os.system(command)
		else:
			LOGGER.info(f"Error moving file in setting {curr_train_setting} with name={abs_filename}")
			LOGGER.info(f"\t\tError moving file in setting {curr_train_setting}")
			# embed()
	



def plot_score_dist_5_CrossEnc(data_name, score_mat_dir):
	"""
	Plot crossencoder score distribution
	:param data_name:
	:param score_mat_dir:
	:return:
	"""
	top_k = 100
	res_dirs = {
			"00_BiencHardNegsAll10"			: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_wo_dp_bs_16_w_hard_bienc",
			"00_BiencHardNegsAll"			: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc",
			"01_BiencHardNegs"				: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc_small_train",
			"02_PathTrainCE"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train",
			# "03_BiencTrain~PathTrainCE"		: "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train_finetune_31",
			"04_PathTrainMargin"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=margin_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train",
			"05_GraphRankCE"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train",
			"06_GraphRankMargin"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train",
			"07_GraphRankCEAll"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf",
			"08_GraphRankMarginAll"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf",
			"09_GraphRankMarginAllRandBienc": "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_rand_bienc",
			"10_GraphRankMarginAllBienc"	: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_bienc",
			"11_BiencHardNegswRankMarginAll": "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=hard_negs_w_rank_s=1234_10_maxnbrs_63_negs_w_bienc",
		}
	
	all_out_files = {}
	for curr_train_setting in res_dirs:
		curr_res_dir = f"{BASE_RES_DIR}/{res_dirs[curr_train_setting]}/{score_mat_dir}"
		LOGGER.info(f"Processing dir {curr_train_setting} --> {curr_res_dir}")
	
		DATASETS = get_dataset_info(data_dir=DATA_DIR, res_dir=curr_res_dir, worlds=WORLDS)
		matrix_file = DATASETS[data_name]["crossenc_ment_to_ent_scores"]
		
		if not os.path.exists(matrix_file):
			LOGGER.info(f"Skipping {curr_res_dir} as matrix file does not exist\n")
			continue
		
		
		with open(matrix_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			
		n_ment, n_ent = crossenc_ment_to_ent_scores.shape
		
		
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		
		title =  f"CrossEnc_scores_for_{data_name}_with_{n_ment}_mentions_and_{n_ent}_entities"
		out_filename = f"{curr_res_dir}/{data_name}/scores/score_dist.pdf"
		visualize_hist(val_matrix=crossenc_ment_to_ent_scores,
					   title=title,
					   topk=top_k,
					   gt_labels=gt_labels,
					   out_filename=out_filename)

		all_out_files[curr_train_setting] = out_filename
		
	pooled_res_dir = f"{BASE_RES_DIR}/5_CrossEnc/PooledResults/{score_mat_dir}/score_dist/{data_name}"
	move_files(pooled_res_dir=pooled_res_dir, all_out_files=all_out_files)
	

def get_all_recall_files_5_CrossEnc(data_name):
	res_dirs = {
			"00_BiencHardNegsAll10"			: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_wo_dp_bs_16_w_hard_bienc",
			"01_BiencHardNegsAll"			: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc",
			"02_BiencHardNegs"				: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc_small_train",
			"03_TFIDFHardNegs"				: "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_tfidf_small_train",
			"04_PathTrainCE"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train",
			# "03_BiencTrain~PathTrainCE"		: "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train_finetune_31",
			"05_PathTrainMargin"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=margin_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train",
			"06_GraphRankCE"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train",
			"07_GraphRankMargin"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train",
			"08_GraphRankCEAll"				: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf",
			"09_GraphRankMarginAll"			: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf",
			"10_GraphRankMarginAllRandBienc": "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_rand_bienc",
			"11_GraphRankMarginAllBienc"	: "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_bienc",
			"12_BiencHardNegswRankMarginAll": "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=hard_negs_w_rank_s=1234_10_maxnbrs_63_negs_w_bienc",
		}

	score_mat_dirs = ["score_mats", "score_mats_wrt_final_model"]
	topk_vals = [1, 100]
	for score_mat_dir, topk in itertools.product(score_mat_dirs, topk_vals):
		for color_grad_variable in ["max_nbrs", "beamsize", ""]:
			for encoder_type in ["crossenc", "bienc"]:
				LOGGER.info(f"Moving recall wrt exact files (with smart init) for {score_mat_dir}")
				all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/nsw/nsw_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/recall_wrt_exact_vs_budget_xlog{color_grad_variable}.pdf"
								 for train_setting, curr_res_dir in res_dirs.items()}
				pooled_res_dir = f"{BASE_RES_DIR}/5_CrossEnc/PooledResults/{score_mat_dir}/recall_wrt_exact_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
				move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
				
				LOGGER.info(f"Moving acc vs budget files for {score_mat_dir}")
				all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/nsw/nsw_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/acc_vs_budget_xlog{color_grad_variable}.pdf"
								 for train_setting, curr_res_dir in res_dirs.items()}
				pooled_res_dir = f"{BASE_RES_DIR}/5_CrossEnc/PooledResults/{score_mat_dir}/acc_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
				move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
				
				LOGGER.info(f"Moving recall vs budget files for {score_mat_dir}")
				all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/nsw/nsw_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/recall_vs_budget_xlog{color_grad_variable}.pdf"
								 for train_setting, curr_res_dir in res_dirs.items()}
				pooled_res_dir = f"{BASE_RES_DIR}/5_CrossEnc/PooledResults/{score_mat_dir}/recall_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
				move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
			
		LOGGER.info(f"Moving score dist files for {score_mat_dir}")
		all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/scores/score_dist.pdf"
						 for train_setting, curr_res_dir in res_dirs.items()}
		pooled_res_dir = f"{BASE_RES_DIR}/5_CrossEnc/PooledResults/{score_mat_dir}/score_dist/{data_name}"
		move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)

	
def plot_score_dist(data_name):
	"""
	Plot crossencoder score distribution
	:param data_name:
	:return:
	"""
	top_k = 100
	res_dirs = {
		# "00_BiencNegs_best_wrt_dev"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats",
		# "00_BiencNegs_2-last"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats_2-last.ckpt",
		# "01_BiencNegs_w_Rerank"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_negs_w_ddp/",
		# "02_BiencNSW_Negs_2_10_250_8K"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_2_bs_10_max_nbrs_250_budget_w_ddp/score_mats_model-1-8359.0-1.14.ckpt",
		# "02_BiencNSW_Negs_2_10_250_11K"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_2_bs_10_max_nbrs_250_budget_w_ddp/score_mats_model-1-11159.0-1.15.ckpt",
		# "03_BiencNSW_Negs_5_10_500"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_5_bs_10_max_nbrs_500_budget_w_ddp/score_mats_model-1-11959.0-1.22.ckpt",
	
		# "04_BiencNSW_Negs_5_10_500_cls_w_lin_0_last"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_0-last.ckpt",
		# ),
		# "04_BiencNSW_Negs_5_10_500_cls_w_lin_best_ckpt"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_model-1-10959.0--78.85.ckpt",
		# ),
		# "05_Rank_CE_KNN_Negs"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt",
		# ),
		# "06_Rank_Margin_KNN_Negs"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/score_mats_model-2-17239.0-1.22.ckpt",
		# ),
		# "08_BiToCrossDistill": (
		# 	"6_ReprCrossEnc/d=ent_link/distill/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only/score_mats_25-last.ckpt",
		# ),
		# "08_BiToCrossDistill_w_6_49_init": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only_w_6_49_init/score_mats_19-last.ckpt",
		# ),
		# "08_BiToCrossDistill_w_mse": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=mse_neg=bienc_distill_s=1234_trn_pro_only/score_mats_17-last.ckpt",
		# ),
		# "08_BiToCrossDistill_w_mse_w_6_49_init": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=mse_neg=bienc_distill_s=1234_trn_pro_only_w_6_49_init/score_mats_17-last.ckpt",
		# ),
		
		# "09_E-CrossEnc": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_small_train/score_mats_model-1-3999.0--77.67.ckpt",
		# ),
		
		"Temp": (
			"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp_w_best_wrt_dev_mrr_cls_w_lin/score_mats_model-1-11359.0--80.19.ckpt",
		),
		# "09_E-CrossEncSmall": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_small_train/score_mats_model-1-3999.0--77.67.ckpt",
		# ),
		# "09_E-CrossEncProWOnly": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_trn_pro_only/score_mats_37-last.ckpt",
		# ),
		# "09_BiToE-CrossEncDistill": (
		# 		"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_distill_s=1234_w_crossenc_w_embeds_trn_pro_only/score_mats_24-last.ckpt",
		# ),
		}

	use_precomp_scores = True
	for curr_train_setting in res_dirs:
		curr_res_dir = f"{BASE_RES_DIR}/{res_dirs[curr_train_setting][0]}"
		LOGGER.info(f"Processing dir {curr_train_setting} --> {curr_res_dir}")
	
		DATASETS = get_dataset_info(data_dir=DATA_DIR, res_dir=curr_res_dir, worlds=WORLDS)
		if use_precomp_scores:
			matrix_file = DATASETS[data_name]["crossenc_ment_to_ent_scores"]

			if not os.path.exists(matrix_file):
				LOGGER.info(f"Skipping {curr_res_dir} as matrix file does not exist\n")
				continue


			with open(matrix_file, "rb") as fin:
				dump_dict = pickle.load(fin)
				crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				test_data = dump_dict["test_data"]
				mention_tokens_list = dump_dict["mention_tokens_list"]
				entity_id_list = dump_dict["entity_id_list"]
			
			n_ment, n_ent = crossenc_ment_to_ent_scores.shape
		else:
			data_file = DATASETS[data_name]["crossenc_ment_and_ent_embeds"]
			if not os.path.exists(data_file):
				LOGGER.info(f"Skipping {curr_res_dir} as data file does not exist\n")
				continue
				
			LOGGER.info("Loading precomputed ment_to_ent embeds")
			with open(data_file, "rb") as fin:
				dump_dict = pickle.load(fin)
	
				all_label_embeds = dump_dict["all_label_embeds"]
				all_input_embeds = dump_dict["all_input_embeds"]
				test_data = dump_dict["test_data"]
				mention_tokens_list = dump_dict["mention_tokens_list"]
				entity_id_list = dump_dict["entity_id_list"]
				entity_tokens_list = dump_dict["entity_tokens_list"]
			
			LOGGER.info("Finished loading")
			################################################################################################################
			
			n_ment, n_ent, embed_dim = all_input_embeds.shape
			
			crossenc_ment_to_ent_scores = torch.nn.CosineSimilarity(dim=-1)(all_input_embeds, all_label_embeds)
			LOGGER.info("Computed scores using precomputed embeddings")
		
		
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		
		title =  f"CrossEnc_scores_for_{data_name}_with_{n_ment}_mentions_and_{n_ent}_entities"
		out_filename = f"{curr_res_dir}/{data_name}/scores/score_dist.pdf"
		visualize_hist(val_matrix=crossenc_ment_to_ent_scores,
					   title=title,
					   topk=top_k,
					   gt_labels=gt_labels,
					   out_filename=out_filename)
		
		# torch_softmax = torch.nn.Softmax(dim=-1)
		# crossenc_ment_to_ent_scores = torch_softmax(crossenc_ment_to_ent_scores)
		# title =  f"Softmax CrossEnc_scores_for_{data_name}_with_{n_ment}_mentions_and_{n_ent}_entities"
		# out_filename = f"{curr_res_dir}/{data_name}/scores/score_dist_softmax.pdf"
		# visualize_hist(val_matrix=crossenc_ment_to_ent_scores,
		# 			   title=title,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels,
		# 			   out_filename=out_filename)
		
		
def get_all_recall_files(data_name):
	curr_exp_dir = "6_ReprCrossEnc"
	res_dirs = {
		# "00_BiencNegs-0-last"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_negs_w_ddp/score_mats_0-last.ckpt",
		# "00_BiencNegs_best_wrt_dev"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats",
		# # "00_BiencNegs_2-last"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats_2-last.ckpt",
		# # "01_BiencNegs_w_Rerank"		: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_negs_w_ddp/",
		# # "02_BiencNSW_Negs_2_10_250_8K"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_2_bs_10_max_nbrs_250_budget_w_ddp/score_mats_model-1-8359.0-1.14.ckpt",
		# "02_BiencNSW_Negs_2_10_250_11K"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_2_bs_10_max_nbrs_250_budget_w_ddp/score_mats_model-1-11159.0-1.15.ckpt",
		# "03_BiencNSW_Negs_5_10_500"	: "6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_5_bs_10_max_nbrs_500_budget_w_ddp/score_mats_model-1-11959.0-1.22.ckpt",
		
		
		# "00_BiencNegs_best_wrt_dev"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp_w_best_wrt_dev_mrr_cls_w_lin/score_mats_model-1-11359.0--80.19.ckpt",
		# 	[""]
		# ),
		# # "04_BiencNSW_Negs_5_10_500_cls_w_lin"	: (
		# # 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_0-last.ckpt",
		# # 	[
		# # 		"std_bienc", "std_bienc_all_ments", "std_bienc_all_ments_l2",
		# # 		"std_bienc_w_anchor_embeds",
		# # 		"ent_only_small_distill_from_std_bienc",
		# # 		"small_distill_from_std_bienc", "distill_from_std_bienc",
		# # 		"small_distill_from_round1_bienc", "distill_from_round1_bienc"
		# # 	]
		# # ),
		# "04_BiencNSW_Negs_5_10_500_cls_w_lin_best_ckpt"	: (
		# 		"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_model-1-10959.0--78.85.ckpt",
		# 		[""]
		# ),
		# "05_Rank_CE_KNN_Negs"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt",
		# 	["", "ent_only_small_distill_from_std_bienc", "small_distill_from_std_bienc",
		# 	 "std_bienc_all_ments_ip", "std_bienc_all_ments_l2"]
		# ),
		# "06_Rank_Margin_KNN_Negs"	: (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/score_mats_model-2-17239.0-1.22.ckpt",
		# 	["", "std_bienc", "ent_distill_w_mse"]
		# ),
		# "07_BiencRerank500": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_500_negs_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_model-1-12279.0--79.47.ckpt",
		# 	[""]
		# ),
		
		#  For score histogram
		"08_BiToCrossDistill": (
			"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only/score_mats_25-last.ckpt",
			[]
		),
		# "08_BiToCrossDistill_w_6_49_init": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only_w_6_49_init/score_mats_19-last.ckpt",
		# 	[]
		# ),
		# "08_BiToCrossDistill_w_mse": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=mse_neg=bienc_distill_s=1234_trn_pro_only/score_mats_17-last.ckpt",
		# 	[]
		# ),
		# "08_BiToCrossDistill_w_mse_w_6_49_init": (
		# 	"6_ReprCrossEnc/d=ent_link/m=cross_enc_l=mse_neg=bienc_distill_s=1234_trn_pro_only_w_6_49_init/score_mats_17-last.ckpt",
		# 	[]
		# )
	}

 
	#
	# topk_vals = [1, 64, 100]
	# for topk in topk_vals:
	# 	for color_grad_variable in ["max_nbrs", "beamsize", ""]:
	# 		encoder_type = "crossenc"
	# 		# for encoder_type in ["crossenc", "bienc"]:
	# 		all_out_files = {}
	# 		for train_setting, (curr_res_dir, bienc_type_vals) in res_dirs.items():
	# 			for bienc_type in bienc_type_vals:
	# 				bienc_type_temp = "small_distill_from_std_bienc_last"if bienc_type == "last_of_small_distill_from_std_bienc" else bienc_type
	# 				bienc_type = "_" + bienc_type if bienc_type != "" else ""
	# 				LOGGER.info(f"Moving recall wrt exact files (with smart init) for topk={topk}, color_grad={color_grad_variable}")
	# 				all_out_files[f"{train_setting}_{bienc_type_temp}"]  = f"{BASE_RES_DIR}/{curr_res_dir}/{data_name}/nsw/search_plots/nsw_recall_wrt_exact{bienc_type}/{encoder_type}_k={topk}/recall_wrt_exact_vs_budget_xlog{color_grad_variable}.pdf"
	#
	# 		pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/recall_wrt_exact_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
	# 		move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
	#
	# 		# all_out_files = {f"{train_setting}":f"{BASE_RES_DIR}/{curr_res_dir}/{data_name}/nsw/search_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/recall_wrt_exact_vs_budget_xlog{color_grad_variable}.pdf"
	# 		# 				 for train_setting, curr_res_dir in res_dirs.items()}
	# 		# pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/recall_wrt_exact_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
	# 		# move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
	#
	# 		# LOGGER.info(f"Moving acc vs budget files for {score_mat_dir}")
	# 		# all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/nsw/search_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/acc_vs_budget_xlog{color_grad_variable}.pdf"
	# 		# 				 for train_setting, curr_res_dir in res_dirs.items()}
	# 		# pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/{score_mat_dir}/acc_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
	# 		# move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
	# 		#
	# 		# LOGGER.info(f"Moving recall vs budget files for {score_mat_dir}")
	# 		# all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{score_mat_dir}/{data_name}/nsw/search_plots/nsw_recall_wrt_exact/{encoder_type}_k={topk}/recall_vs_budget_xlog{color_grad_variable}.pdf"
	# 		# 				 for train_setting, curr_res_dir in res_dirs.items()}
	# 		# pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/{score_mat_dir}/recall_vs_budget/{encoder_type}_k={topk}/{data_name}/{color_grad_variable}"
	# 		# move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)
	#
	# # Moving avg distance b/w entity plots
	# LOGGER.info("Moving avg distance plots")
	# all_out_files_1 = {}
	# all_out_files_2 = {}
	# for train_setting, (curr_res_dir, bienc_type_vals) in res_dirs.items():
	# 	for bienc_type in bienc_type_vals:
	# 		# bienc_type_temp = bienc_type if bienc_type != "last_of_small_distill_from_std_bienc" else "small_distill_from_std_bienc_last"
	# 		# bienc_type = "_" + bienc_type if bienc_type != "" else ""
	# 		bienc_type = ""
	# 		all_out_files_1[f"{train_setting}_{bienc_type}"] = f"{BASE_RES_DIR}/{curr_res_dir}/{data_name}/nsw/graph_analysis/plots/crossenc_k=100_max_nbrs_10{bienc_type}/within_type_path_len_vs_num_ents_mst.pdf"
	# 		all_out_files_2[f"{train_setting}_{bienc_type}"] = f"{BASE_RES_DIR}/{curr_res_dir}/{data_name}/nsw/graph_analysis/plots/crossenc_k=100_max_nbrs_10{bienc_type}/cross_type_path_len_vs_num_ents_mst.pdf"
	#
	# pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/graph_analysis/crossenc_k=100_max_nbrs_10_within/{data_name}"
	# move_files(all_out_files=all_out_files_1, pooled_res_dir=pooled_res_dir)
	#
	# pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/graph_analysis/crossenc_k=100_max_nbrs_10_cross/{data_name}"
	# move_files(all_out_files=all_out_files_2, pooled_res_dir=pooled_res_dir)
	
	
	LOGGER.info(f"Moving score dist files")
	all_out_files = {train_setting:f"{BASE_RES_DIR}/{curr_res_dir}/{data_name}/scores/score_dist.pdf"
					 for train_setting, (curr_res_dir,_) in res_dirs.items()}
	pooled_res_dir = f"{BASE_RES_DIR}/{curr_exp_dir}/PooledResults/score_dist/{data_name}"
	move_files(all_out_files=all_out_files, pooled_res_dir=pooled_res_dir)


def main():
	
	
	
	# parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	# parser.add_argument("--data_name", type=str, choices=[w for _,w in WORLDS] + ["all"], help="Dataset name")
	# parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	
	# args = parser.parse_args()
	# data_name = args.data_name
	# res_dir = args.res_dir
	# Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	
	# all_data_names = ["lego", "pro_wrestling", "doctor_who", "american_football"]
	all_data_names = ["lego", "pro_wrestling"]
	for data_name in tqdm(all_data_names):
		LOGGER.info(f"Debugging models for world = {data_name}")
		# score_mat_dirs = ["score_mats", "score_mats_wrt_final_model"]
		# for score_mat_dir in score_mat_dirs:
		# 	plot_score_dist_5_CrossEnc(data_name=data_name, score_mat_dir=score_mat_dir)
		# get_all_recall_files_5_CrossEnc(data_name=data_name)
		
		plot_score_dist(data_name=data_name)
		# get_all_recall_files(data_name=data_name)
		


if __name__ == "__main__":
	main()

