import os
import sys
import json
import logging
import itertools
from pathlib import Path

from time import gmtime, strftime
from utils.zeshel_utils import N_ENTS_ZESHEL as NUM_ENTS

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _get_param_config(base_res_dir):
	data_name_vals = ["pro_wrestling", "lego", "doctor_who"]
	# data_name_vals = ["doctor_who"]
	data_name_vals = ["pro_wrestling", "lego"]
	data_name_vals = ["pro_wrestling", "lego", "yugioh"]
	data_name_vals = ["pro_wrestling", "doctor_who"]
	data_name_vals = ["pro_wrestling"]
	# data_name_vals = ["lego"]
	# data_name_vals = ["yugioh"]
	
	score_mat_dir_vals = {
		# "00_Rank_Margin_6_117": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/score_mats_model-2-17239.0-1.22.ckpt",
		# "01_Rank_CE_6_93": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt",
		
		"02_CE_0_6_400": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/score_mats_eoe-0-last.ckpt",
		
		# "02_CE_6_49": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp_w_best_wrt_dev_mrr_cls_w_lin/score_mats_model-1-11359.0--80.19.ckpt",
		
		# "03_CE_0_last_6_82": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_0-last.ckpt",
		# "04_Distill": "../../results/6_ReprCrossEnc/d=ent_link/distill/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only/score_mats_25-last.ckpt",
		# "05_E-CrossEnc_Small_6_248": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_small_train/score_mats_model-1-3999.0--77.67.ckpt",
		
		# "05_E-CrossEnc_6_256": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt",
		
		# "06_E-CrossEnc_Rank_CE_6_260": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-1-12279.0-3.19.ckpt",
		
		# "07_E-Joint_6_280":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_crossenc_w_embeds_w_0.5_bi_cross_loss/score_mats_model-1-12279.0--78.92.ckpt",
		# "07_CLS-Joint_6_282":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_cls_crossenc_w_0.5_bi_cross_loss/score_mats_model-2-18439.0--79.21.ckpt",
		
		# "08_E-Joint_S_6_293":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-19679.0--77.23.ckpt",
		# "08_CLS-Joint_S_6_296":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_crossenc_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-20919.0--80.86.ckpt",
		# "08_CLS-Joint_S_6_380":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_crossenc_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-22159.0--78.55.ckpt",
	}
	
	bi_model_file_vals = {
		# "01_Distill_7_5": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_all_data/model/model-3-12318.0-1.92.ckpt",
		"00_HardNegs_6_20": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt",
		
		"00_Distill_7_71": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-19-last.ckpt",
		"00_Distill_S_7_73": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-19-last.ckpt",
		
		"00_Distill_CE_as_Pos_7_85": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-4-last.ckpt",
		"00_Distill_CE_as_Pos_S_7_87": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-3-last.ckpt",
		

		# "02_HardNegs_Shared_CLS_6_276": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_shared_params_n_cls_pool/model/model-3-12318.0-2.04.ckpt",
		# "02_HardNegs_Shared_Spk-Tkn_6_286": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_shared_params_n_spl_tkn_pool/model/model-3-12239.0-2.15.ckpt",
		#
		# "07_E-Joint_6_280":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_crossenc_w_embeds_w_0.5_bi_cross_loss/model/model-1-12279.0--78.92.ckpt",
		# "07_CLS-Joint_6_282":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_cls_crossenc_w_0.5_bi_cross_loss/model/model-2-18439.0--79.21.ckpt",
		#
		# "08_E-Joint_S_6_293"  : "../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/model/model-3-19679.0--77.23.ckpt",
		# "08_CLS-Joint_S_6_296": "../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_crossenc_w_0.5_bi_cross_loss_from_scratch/model/model-3-20919.0--80.86.ckpt",
		# "08_CLS-Joint_S_6_380": "../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_crossenc_w_0.5_bi_cross_loss_from_scratch/model/model-3-22159.0--78.55.ckpt",
		
	}
	
	ent_model_file_vals = {
		"":"None",
		# "00_Distill_7_71": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-19-last.ckpt",
		# "00_Distill_S_7_73": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-19-last.ckpt",
		
		# "00_Distill_CE_as_Pos_7_85": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-4-last.ckpt",
		# "00_Distill_CE_as_Pos_S_7_87": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-3-last.ckpt",
	}
	
	
	# graph_metric_vals = ["l2", "ip"]
	graph_metric_vals = ["l2"]
	# graph_type_vals = ["nsw", "knn", "knn_e2e"]
	graph_type_vals = ["nsw"]
	# graph_type_vals = ["nsw"]
	# graph_type_vals = ["knn_e2e"]
	# graph_type_vals = ["nsw_e2e"]
	
	# embed_type_vals = ["bienc", "tfidf", "anchor", "none"]
	# embed_type_vals = ["bienc", "anchor", "none", "c-anchor"]
	# embed_type_vals = ["bienc", "anchor", "none"]
	# embed_type_vals = ["bienc", "anchor"]
	embed_type_vals = ["bienc", "ent", "anchor"]
	
	# embed_type_vals = ["none"]
	# embed_type_vals = ["c-anchor"]
	# embed_type_vals = ["bienc", "tfidf", "none"]
	# embed_type_vals = ["bienc", "none"]
	entry_method_vals = ["bienc"]
	# entry_method_vals = ["bienc"]
	n_ment_vals = [100]
	# n_ment_vals = [-1, 100]
	
	# a2e_suffix_vals = ["", "_bienc_cluster", "_all_m2e_anchor_cluster"]
	# a2e_suffix_vals = ["", "_bienc_cluster", "_all_m2e_anchor_cluster", "_100_m2e_anchor_cluster"]
	# a2e_suffix_vals = ["", "topk_1000_embed_bienc_m2e_kmed_cluster_alt", "topk_1000_embed_tfidf_m2e_kmed_cluster_alt", "topk_500_embed_bienc_m2e_kmed_cluster_alt"]
	a2e_suffix_vals = [""]
	# masked_node_frac_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	masked_node_frac_vals = [0.0]
	# masked_node_frac_vals = [1.0]
	
	all_run_configs = []
	for data_name, score_mat_key, bi_model_key, ent_model_key, graph_type, graph_metric, embed_type, entry_method, n_ment, a2e_suffix, masked_node_frac in \
			itertools.product(data_name_vals, score_mat_dir_vals, bi_model_file_vals, ent_model_file_vals, graph_type_vals, graph_metric_vals, embed_type_vals, entry_method_vals, n_ment_vals, a2e_suffix_vals, masked_node_frac_vals):
		
		res_dir = f"{base_res_dir}/{data_name}"
		res_dir = f"{base_res_dir}/{data_name}_w_distill" # FIXME: Remove this
		
		e2e_score_filename = f"{score_mat_dir_vals[score_mat_key]}/{data_name}/"
		e2e_score_filename += f"ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_topk_100_embed_bienc_m2e_.pkl"
		
		a2e_score_filename = f"{score_mat_dir_vals[score_mat_key]}/{data_name}/"
		a2e_score_filename += f"ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_{a2e_suffix}.pkl"
		
		# misc = f"c={score_mat_key}_b={bi_model_key}_n_m={n_ment}{a2e_suffix}_mnf={masked_node_frac}"
		# misc = f"c={score_mat_key}_b={bi_model_key}_n_m={n_ment}{a2e_suffix}_w_ment_filter"
		misc = f"c={score_mat_key}_b={bi_model_key}_n_m={n_ment}{a2e_suffix}{ent_model_key}"
		
		
		curr_run_params = {
			"project_name": "Graph_Search",
			"data_name":data_name,
			"n_ment": n_ment,
			"embed_type": embed_type,
			"entry_method": entry_method,
			"graph_type": graph_type,
			"graph_metric": graph_metric,
			"masked_node_frac": masked_node_frac,
			"misc": misc,
			"bi_model_file": bi_model_file_vals[bi_model_key],
			"entity_model_file_for_index": ent_model_file_vals[ent_model_key],
			"e2e_score_filename": e2e_score_filename,
			"a2e_score_filename": a2e_score_filename,
			"score_mat_dir": score_mat_dir_vals[score_mat_key],
			"res_dir": res_dir,
		}
	
		# # l2 not supported w/ knn
		# if graph_type == "knn" and graph_metric == "l2": continue

		# graph_metric ip or l2 not used for knn_e2e so skipping one of them
		if graph_type in ["knn_e2e", "nsw_e2e"] and graph_metric == "ip": continue
		
		# embed_type none only works with knn_e2e and vice-versa
		if embed_type == "none" and graph_type not in ["knn_e2e", "nsw_e2e"]: continue
		if embed_type != "none" and graph_type in ["knn_e2e", "nsw_e2e"]: continue
		
		# For embed_type other than c-anchor, a2e_suffix vals other than "" don't make any difference
		# For embed_type c-anchor, a2e_suffix val = "" does not make sense
		if embed_type != "c-anchor" and a2e_suffix != "": continue
		if embed_type == "c-anchor" and a2e_suffix == "": continue
		
		# Embed_type = ent should be accompanied with non-empty ent_model_key, and vice-versa
		if embed_type == "ent" and ent_model_key == "": continue
		if embed_type != "ent" and ent_model_key != "": continue
		
		
		all_run_configs += [curr_run_params]
	
	LOGGER.info(f"Returning {len(all_run_configs)} configs")
	return all_run_configs


def launch_jobs():
	
	base_res_dir = "../../results/6_ReprCrossEnc/Graph_Search"
	Path(base_res_dir).mkdir(exist_ok=True, parents=True)
	
	all_configs = _get_param_config(base_res_dir=base_res_dir)
	
	all_commands = []
	previously_run_commands = []
	found = 0
	for ctr, curr_config in enumerate(all_configs):
		curr_command = f"sbatch -p gpu --gres gpu:1 --mem 32GB --job-name gs_{curr_config['data_name']}_{ctr} --exclude gpu-0-[0-1] bin/run.sh python eval/nsw_eval_zeshel.py "
		# curr_command = f"sbatch -p cpu --mem 32GB --job-name gs_{curr_config['data_name']}_{ctr} bin/run.sh python eval/nsw_eval_zeshel.py "
		for key,val in curr_config.items():
			curr_command += f" --{key} {val} "
		
		curr_command += " --force_exact_init_search 1 "
		res_file = "{res_dir}/{data_name}/{graph_type}/search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}_{misc}.json".format(
			res_dir=curr_config["res_dir"],
			data_name=curr_config["data_name"],
			graph_type=curr_config["graph_type"],
			graph_metric=curr_config["graph_metric"],
			embed_type=curr_config["embed_type"],
			entry_method=curr_config["entry_method"],
			misc=curr_config["misc"],
		)
		
		launch_new_jobs = int(sys.argv[1])
		launch_jobs_even_if_already_run = int(sys.argv[2])
		if launch_jobs_even_if_already_run or not os.path.isfile(res_file):
			all_commands += [curr_command]
			if launch_new_jobs: os.system(command=curr_command)
			# LOGGER.info(f"command: {curr_command}")
		else:
			previously_run_commands += [curr_command]
	
	LOGGER.info(f"Previously run commands = {len(previously_run_commands)}")
	LOGGER.info(f"Commands run now = {len(all_commands)}")
	LOGGER.info(f"Found = {found}")
	

if __name__ == "__main__":
	launch_jobs()
