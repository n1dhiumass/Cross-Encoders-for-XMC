import os
import sys
import json
import glob
import logging
import itertools
import numpy as np
from tqdm import tqdm
from IPython import embed
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from textwrap import wrap

from utils.zeshel_utils import N_ENTS_ZESHEL as NUM_ENTS

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)
cmap = [
	('firebrick', 'lightsalmon'),
	('green', 'yellowgreen'),
	('navy', 'skyblue'),
	('darkorange', 'gold'),
	('deeppink', 'violet'),
	('olive', 'y'),
	('darkviolet', 'orchid'),
	('deepskyblue', 'lightskyblue'),
	('sienna', 'tan'),
	('gray', 'silver')
]


def _str_to_tuple(tuple_str, dtype):
	
	tuple_str = tuple_str[1:-1] # Remove braces
	ans_tuple  = [dtype(x) for x in tuple_str.split(",")]
	ans_tuple = tuple(ans_tuple)
	return ans_tuple



def _get_cost_bucket(cost_val, all_cost_bkts):
	
	for lower,upper in all_cost_bkts:
		if (lower < cost_val) and (cost_val <= upper):
			return lower, upper
	
	return None, None


def process_res_file(file, comp_budgets):
	"""
	
	:param file:
	:return: Dict with following structure
				topk -> Dict with that maps
						"all" -> Dict with that maps
								cost_bucket -> All {cost:<actual_cost>, recall:<>, key:<hypeparam_str>} dicts
												that have cost within cost_bucket
						"best" -> Dict with that maps
								cost_bucket -> BEST recall {cost:<actual_cost>, recall:<>, key:<hypeparam_str>} of all dicts
												that have cost within cost_bucket
	"""
	try:
		if not os.path.isfile(file): return {}
		
		with open(file, "r") as fin:
			data = json.load(fin)
	
		keys_to_ignore = ["data_info", "arg_dict", "other_args"]
		hyperparams = data["arg_dict"]
		graph_type = hyperparams["graph_type"]
		embed_type = hyperparams["embed_type"]
		graph_metric = hyperparams["graph_metric"]
		entry_method = hyperparams["entry_method"]
		misc = hyperparams["misc"]
		n_ment = hyperparams["n_ment"]
		
		
		cost_bkts = list(zip(comp_budgets[:-1], comp_budgets[1:]))
		
	
		bktd_cost_vs_recall = defaultdict(lambda :{"all":defaultdict(list), "best":{}, "init":{}})
		
		for key in data:
			if key in keys_to_ignore: continue
			topk = int(key.split("_")[0][2:])
			# Example key= "k=64_b=64_init_bienc_budget=64_max_nbrs=10"
			# topk=1
			# beamsize = 1
			# entry_method = "bienc"
			# comp_budget = 64
			# max_nbrs = 10
			
			# # # FIXME: Remove following if condition - this is to filter data for beamsize=5 and max_nbrs=10
			# if "_budget=0_" not in key:
			# 	if not key.endswith("max_nbrs=10"): continue
			# 	if "_b=5_" not in key: continue

			
			recall = float(data[key][f"crossenc~exact_vs_{graph_type}~common_frac"][0][5:])
			actual_cost = data[key][f"crossenc~{graph_type}~num_score_comps~p50"]
	
			cost_bkt = _get_cost_bucket(cost_val=actual_cost, all_cost_bkts=cost_bkts)
			if cost_bkt == (None, None): continue # Skipping anything outside of defined buckets
			
			cost_bkt_str = str(cost_bkt)
			bktd_cost_vs_recall[topk]["all"][cost_bkt_str] += [{"cost":actual_cost, "recall":recall, "key":key}]
		
		# Find best recall value in each cost bucket
		for topk in bktd_cost_vs_recall:
			for curr_cost_bkt_str in bktd_cost_vs_recall[topk]["all"]:
				all_ans = bktd_cost_vs_recall[topk]["all"][curr_cost_bkt_str]
				
				# Filter out results for initial retrieval from graph search ones
				graph_search_ans = [curr_ans for curr_ans in all_ans if "budget=0" not in curr_ans["key"]]
				init_ans = [curr_ans for curr_ans in all_ans if "budget=0" in curr_ans["key"]]
				
				# Sort them by recall and pick best one
				graph_search_ans_sorted = sorted(graph_search_ans, key=lambda x:x["recall"], reverse=True)
				init_ans_sorted = sorted(init_ans, key=lambda x:x["recall"], reverse=True)
				
				if len(graph_search_ans_sorted) > 0: # If there is at least one point in this cost bucket
					bktd_cost_vs_recall[topk]["best"][curr_cost_bkt_str] = graph_search_ans_sorted[0]
				
				if len(init_ans_sorted) > 0: # If there is at least one point in this cost bucket
					bktd_cost_vs_recall[topk]["init"][curr_cost_bkt_str] = init_ans_sorted[0]
				
				# #TODO: Maybe also calculate avg/median performance in this bucket?
				# bktd_cost_vs_recall["avg"][curr_cost_bkt] = ??
				pass
			
	
		return bktd_cost_vs_recall
	except Exception as e:
		embed()
		raise e



def process_res_for_rq(data, data_name, fixed_params, var_params):
	"""
	
	:param data: Dict with following structure
				str_with_uniq_val_for_each_parameter to dict that maps
					topk -> Dict with that maps
							"all" -> Dict with that maps
									cost_bucket -> All {cost:<actual_cost>, recall:<>, key:<hypeparam_str>} dicts
													that have cost within cost_bucket
							"best" -> Dict with that maps
									cost_bucket -> BEST recall {cost:<actual_cost>, recall:<>, key:<hypeparam_str>} of all dicts
													that have cost within cost_bucket
													
		 
	:param data_name:
	:param fixed_params:
	:param var_params:
	:return:
	"""
	# template = "search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}_c={crossenc}_b={bienc}_n_m={n_ment}{a2e_suffix}"
	# template = "search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}_c={crossenc}_b={bienc}_n_m={n_ment}{ent_model}"
	template = "search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}_c={crossenc}_b={bienc}_n_m={n_ment}_mask_type={mask_type}_method={mask_method}_frac={masked_node_frac}"
	# template = "search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}_c={crossenc}_b={bienc}_n_m={n_ment}"
	
	all_param_vals = {
		# "graph_metric": ["l2", "ip"],
		"graph_metric": ["l2"],
		
		# "graph_type": ["nsw", "knn", "knn_e2e"],
		# "graph_type": ["nsw", "knn"],
		# "graph_type": ["nsw", "knn"],
		"graph_type": ["nsw"],
		
		# "embed_type": ["bienc", "tfidf", "anchor", "e2e", "c-anchor"],
		# "embed_type": ["bienc", "anchor", "e2e", "ent"],
		# "embed_type": ["c-anchor", "anchor"],
		# "embed_type": ["e2e"],
		# "embed_type": ["cur-anchor"],
		# "embed_type": ["bienc", "anchor", "e2e"],
		# "embed_type": ["bienc", "cur"],
		"embed_type": ["bienc"],
		
		"entry_method": ["bienc"],
		# "entry_method": ["bienc"],
		
		"n_ment": [100],
		# "n_ment": [100, -1],
		# "n_ment": [-1],
		
		"crossenc": {
			# "00_Rank_Margin_6_117": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/score_mats_model-2-17239.0-1.22.ckpt",
			# "01_Rank_CE_6_93": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt",
			# "06_E-CrossEnc_Rank_CE_6_260": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-1-12279.0-3.19.ckpt",
			# "04_Distill": "../../results/6_ReprCrossEnc/d=ent_link/distill/m=cross_enc_l=ce_neg=bienc_distill_s=1234_trn_pro_only/score_mats_25-last.ckpt",
			# "05_E-CrossEnc_Small_6_248": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_small_train/score_mats_model-1-3999.0--77.67.ckpt",
			# "03_CE_0_last_6_82": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_0-last.ckpt",
			
			# "02_CE_0_6_400": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/score_mats_eoe-0-last.ckpt",
			
			
			"05_E-CrossEnc_6_256": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt",
			# "07_E-Joint_6_280":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_crossenc_w_embeds_w_0.5_bi_cross_loss/score_mats_model-1-12279.0--78.92.ckpt",
			# "08_E-Joint_S_6_293":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-19679.0--77.23.ckpt",
			
			# "02_CE_6_49": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp_w_best_wrt_dev_mrr_cls_w_lin/score_mats_model-1-11359.0--80.19.ckpt",
			# "07_CLS-Joint_6_282":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_cls_crossenc_w_0.5_bi_cross_loss/score_mats_model-2-18439.0--79.21.ckpt",
			# "08_CLS-Joint_S_6_380":"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_crossenc_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-22159.0--78.55.ckpt",
			
			# "08_CLS-Joint_S_6_296":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_crossenc_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-20919.0--80.86.ckpt",
		},
		"bienc" : {
			# "01_Distill_7_5": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_all_data/model/model-3-12318.0-1.92.ckpt",
			"00_HardNegs_6_20": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt",
			
			# "00_Distill_7_71": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-19-last.ckpt",
			# "00_Distill_S_7_73": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-19-last.ckpt",
			#
			# "00_Distill_CE_as_Pos_7_85": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-4-last.ckpt",
			# "00_Distill_CE_as_Pos_S_7_87": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-3-last.ckpt",


			# "02_HardNegs_Shared_CLS_6_276": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_shared_params_n_cls_pool/model/model-3-12318.0-2.04.ckpt",
			# "08_CLS-Joint_S_6_380":"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_crossenc_w_0.5_bi_cross_loss_from_scratch/model/model-3-22159.0--78.55.ckpt",
			#
			# "02_HardNegs_Shared_Spk-Tkn_6_286": "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_shared_params_n_spl_tkn_pool/model/model-3-12239.0-2.15.ckpt",
			# "08_E-Joint_S_6_293"  : "../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/model/model-3-19679.0--77.23.ckpt",
		
			# "07_E-Joint_6_280":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_crossenc_w_embeds_w_0.5_bi_cross_loss/model/model-1-12279.0--78.92.ckpt",
			# "07_CLS-Joint_6_282":"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_6_20_bienc_w_cls_crossenc_w_0.5_bi_cross_loss/model/model-2-18439.0--79.21.ckpt",
			# "08_CLS-Joint_S_6_296": "../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_crossenc_w_0.5_bi_cross_loss_from_scratch/model/model-3-20919.0--80.86.ckpt",
		},
		# "ent_model":{
		# 	"":"",
		# 	"00_Distill_7_71": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-19-last.ckpt",
		# 	"00_Distill_S_7_73": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-19-last.ckpt",
		# 	"00_Distill_CE_as_Pos_7_85": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_6_20/model/eoe-4-last.ckpt",
		# 	"00_Distill_CE_as_Pos_S_7_87": "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=top_ce_as_pos_w_bienc_hard_negs_s=1234_distill_w_64_crossenc_negs_wrt_6_400_pro_from_scratch/model/eoe-3-last.ckpt",
		# },
		# "a2e_suffix" : ["", "_bienc_cluster", "_all_m2e_anchor_cluster", "_100_m2e_anchor_cluster"],
		# "a2e_suffix" : ["", "_bienc_cluster_w_1K_anchors", "_all_m2e_anchor_cluster_w_1K_anchors"],
		# "a2e_suffix" : ["", "_bienc_cluster_w_ment_filter", "_all_m2e_anchor_cluster_w_ment_filter", "_100_m2e_anchor_cluster_w_ment_filter"],
		# "a2e_suffix" : ["", "topk_1000_embed_bienc_m2e_kmed_cluster_alt", "topk_1000_embed_tfidf_m2e_kmed_cluster_alt", "topk_500_embed_bienc_m2e_kmed_cluster_alt"],
		# "a2e_suffix" : [""],
		# "masked_node_frac": [0.5, 1.0],
		# "masked_node_frac": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
		# "masked_node_frac": [0.0, 0.6, 0.9],
		"masked_node_frac": [0.0, 0.9],
		"mask_type": ["adapt_soft"],
		# "mask_type": ["soft", "adapt_soft"],
		# "mask_type": ["adapt_soft"],
		# "mask_method": ["cur_1000", "cur"],
		"mask_method": ["cur", "cur_local"],
		# "mask_method": ["cur"],
		# "masked_node_frac": [0.2, 0.4, 0.6, 0.8, 1.0],
		# "masked_node_frac": [1.0],
	}
	final_res = defaultdict(lambda :defaultdict(dict))
	
	all_fixed_param_vals = [ all_param_vals[param_name] for param_name in fixed_params]
	all_var_param_vals = [ all_param_vals[param_name] for param_name in var_params]
	
	for curr_fixed_param_vals in itertools.product(*all_fixed_param_vals):
		fixed_key = "_".join([ f"{param}={param_val}" for param, param_val in zip(fixed_params, curr_fixed_param_vals)])
		# eg fixed_key = f"graph_metric={graph_metric}_entry_method={entry_method}_crossenc={crossenc}_bienc={bienc}_n_ment={n_ment}"
		for curr_var_param_vals in itertools.product(*all_var_param_vals):
			# eg var_key = f"graph_type={graph_type}_embed_type={embed_type}"
			var_key = "_".join([ f"{param}={param_val}" for param, param_val in zip(var_params, curr_var_param_vals)])
			
			comb_param_dict = {param_name:param_val for param_name, param_val in zip(fixed_params, curr_fixed_param_vals)}
			comb_param_dict.update({param_name:param_val for param_name, param_val in zip(var_params, curr_var_param_vals)})
			comb_key = template.format(
				data_name=data_name,
				**comb_param_dict
			)
			# if comb_param_dict["graph_type"] == "knn" and comb_param_dict["embed_type"] != "e2e": continue # FIXME: Remove this
			if comb_param_dict["masked_node_frac"] == 0.0 and (comb_param_dict["mask_type"] not in ["hard"] or comb_param_dict["mask_method"] != "cur"): continue # FIXME: Remove this
		
				
			if comb_key not in data: continue
			for topk in data[comb_key]:
				final_res[fixed_key][var_key][topk] = {"data":data[comb_key][topk], "AOC":1} # TODO: Calc AOC
		
	
	return final_res


def create_combined_result_file(base_res_dir):
	
	
	comp_budgets = [50, 64, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000]
	comp_budgets = [50, 64, 100, 250, 500, 1000, 2000]
	# comp_budgets = np.arange(50, 2000, 2).tolist()
	# comp_budgets = [300, 400, 500, 600, 800]
	
	
	file_list = glob.glob(f"{base_res_dir}/*/*.json")

	LOGGER.info(f"Processing {len(file_list)} files")
	all_data = {}
	for full_filename in tqdm(file_list):
		rel_filename = full_filename.split("/")[-1]
		rel_filename = rel_filename[:-5] if rel_filename.endswith(".json") else rel_filename

		bktd_cost_vs_recall = process_res_file(file=full_filename, comp_budgets=comp_budgets)

		if len(bktd_cost_vs_recall) == 0: continue

		all_data[rel_filename] = bktd_cost_vs_recall
		# search_eval_pro_wrestling_g=knn_e2e_l2_e=none_init=bienc_c=00_Rank_Margin_6_117_b=00_HardNegs_6_20_n_m=100
		
		for temp_gtype in ["knn", "nsw"]:
			if f"g={temp_gtype}_e2e_l2" in rel_filename:
				# Also duplicate data for graph_type={temp_gtype}_e2e_ip as {temp_gtype}_e2e_ip and {temp_gtype}_e2e_l2 are same because {temp_gtype}_e2e does not use any embedding to build graph
				
				temp_fname = rel_filename.replace(f"g={temp_gtype}_e2e_l2", f"g={temp_gtype}_e2e_ip")
				all_data[temp_fname] = bktd_cost_vs_recall
				
				
				
				for _embed_type in ["bienc", "tfidf", "anchor"]:
					temp_fname = rel_filename.replace(f"g={temp_gtype}_e2e_l2_e=none", f"g={temp_gtype}_e2e_l2_e={_embed_type}")
					all_data[temp_fname] = bktd_cost_vs_recall
					
					temp_fname = rel_filename.replace(f"g={temp_gtype}_e2e_l2_e=none", f"g={temp_gtype}_e2e_ip_e={_embed_type}")
					all_data[temp_fname] = bktd_cost_vs_recall
				
				
				# For plotting to treat ent2ent scores as another entity-embedding option although this is no explicit embedding produced in this method
				temp_fname = rel_filename.replace(f"g={temp_gtype}_e2e_l2_e=none", f"g={temp_gtype}_l2_e=e2e")
				all_data[temp_fname] = bktd_cost_vs_recall
				
				temp_fname = rel_filename.replace(f"g={temp_gtype}_e2e_l2_e=none", f"g={temp_gtype}_ip_e=e2e")
				all_data[temp_fname] = bktd_cost_vs_recall
				

	LOGGER.info(f"Data successfully read for {len(all_data)} files")

	# Saving intermediate data
	inter_fname = f"{base_res_dir}/inter_res.json"
	with open(inter_fname, "w") as fout:
		json.dump(all_data, fout, indent=4)
	
	

def process_results(data_name):
	try:
		
		# base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/{data_name}/{data_name}"
		base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/{data_name}/node_masking/{data_name}" # FIXME: Use for debuggging only
		# base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/{data_name}_w_distill/{data_name}" # FIXME: Use for debuggging only
		# base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/{data_name}_debug_cur/{data_name}" # FIXME: Use for debuggging only
		# base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/
		# {data_name}_masked_nodes/{data_name}"
		
		# LOGGER.info("Skipping creating intermediate file, using existing file")
		create_combined_result_file(base_res_dir=base_res_dir)
		
		# Loading intermediate data
		LOGGER.info(f"Processing data for each research question for {data_name}")
		inter_fname = f"{base_res_dir}/inter_res.json"
		with open(inter_fname, "r") as fin:
			all_data = json.load(fin)
		
		
		RQs = {
			"RQ/1_Graph_Type_And_Entity_Embedding_To_Use": {
				"fixed_params":["graph_metric", "entry_method", "crossenc", "bienc", "n_ment", ],
				# "var_params":["graph_type", "embed_type"],
				"var_params":["graph_type", "embed_type", "masked_node_frac", "mask_type", "mask_method"],
				# "var_params":["graph_type", "embed_type", "ent_model"],
				# "var_params":["graph_type", "embed_type", "a2e_suffix"],
			},
			# "RQ/1a_Graph_Type_For_Search": {
			# 	# "fixed_params":["graph_metric", "embed_type", "entry_method", "crossenc", "bienc", "n_ment", "a2e_suffix"],
			# 	"fixed_params":["graph_metric", "embed_type", "entry_method", "crossenc", "bienc", "n_ment"],
			# 	"var_params":["graph_type"]
			# },
			# "RQ/1b_Entity_Embedding_To_Use": {
			# 	# "fixed_params":["graph_type", "graph_metric", "entry_method", "crossenc", "bienc", "n_ment", "a2e_suffix"],
			# 	"fixed_params":["graph_type", "graph_metric", "entry_method", "crossenc", "bienc", "n_ment"],
			# 	"var_params":["embed_type"]
			# },
			# "RQ/2_Entry_Points_In_Graph": {
			# 	# "fixed_params":["graph_type", "graph_metric", "embed_type", "crossenc", "bienc", "n_ment", "a2e_suffix"],
			# 	# "fixed_params":["graph_type", "graph_metric", "embed_type", "crossenc", "bienc", "n_ment"],
			# 	"fixed_params":["graph_type", "graph_metric", "embed_type", "crossenc", "bienc", "n_ment", "masked_node_frac"],
			# 	"var_params":["entry_method"]
			# },
			# "RQ/3_Node_Similarity": {
			# 	# "fixed_params":["graph_type", "embed_type", "entry_method", "crossenc", "bienc", "n_ment", "a2e_suffix"],
			# 	"fixed_params":["graph_type", "embed_type", "entry_method", "crossenc", "bienc", "n_ment"],
			# 	"var_params":["graph_metric"]
			# },
			# "RQ/4_Training_Method_for_CrossEnc": {
			# 	# "fixed_params":["graph_type", "graph_metric", "embed_type", "entry_method", "bienc", "n_ment", "a2e_suffix"],
			# 	"fixed_params":["graph_type", "graph_metric", "embed_type", "entry_method", "bienc", "n_ment"],
			# 	"var_params":["crossenc"]
			# },
			# "RQ/5_Training_Method_for_Bienc": {
			# 	# "fixed_params":["graph_type", "graph_metric", "embed_type", "entry_method", "crossenc", "n_ment"],
			# 	"fixed_params":["graph_type", "graph_metric", "embed_type", "entry_method", "crossenc", "n_ment", "ent_model"],
			# 	"var_params":["bienc"]
			# }
		}
		
		for curr_rq in RQs:
			LOGGER.info(f"Processing data for RQ : {curr_rq}")
			processed_res = process_res_for_rq(
				data_name=data_name,
				data=all_data,
				fixed_params=RQs[curr_rq]["fixed_params"],
				var_params=RQs[curr_rq]["var_params"],
			)
			# Save data
			process_fname = f"{base_res_dir}/{curr_rq}/processed_res.json"
			Path(os.path.dirname(process_fname)).mkdir(exist_ok=True, parents=True)
			with open(process_fname, "w") as fout:
				json.dump(processed_res, fout, indent=4)
			
			# Plot data
			plot_processed_results(res_fname=process_fname, plot_init_only=False)
			# plot_processed_results(res_fname=process_fname, plot_init_only=True)
			
	except Exception as e:
		embed()
		raise e
	
	
def plot_processed_results(res_fname, plot_init_only):
	
	
	
	base_plt_res_dir = f"{os.path.dirname(res_fname)}/plots_init_only={plot_init_only}"
	Path(base_plt_res_dir).mkdir(exist_ok=True, parents=True)
	with open(res_fname, "r") as fin:
		processed_data = json.load(fin)
	
	# # Top-k vals in data
	# topk_vals = [processed_data[k1][k2].keys() for k1 in processed_data for k2 in processed_data[k1]]
	# topk_vals = set([x for x_list in topk_vals for x in x_list])
	# LOGGER.info(f"Topk_vals in data = {topk_vals}")
	
	# topk_vals = [str(64)] # Remove this to plot for all topk_vals
	topk_vals = [str(1)] # Remove this to plot for all topk_vals
	topk_vals = [str(1), str(64)] # Remove this to plot for all topk_vals
	# topk_vals = [str(64)] # Remove this to plot for all topk_vals
	
	for fixed_key in processed_data:
		var_keys = processed_data[fixed_key].keys()
		for topk in topk_vals:
			plt.clf()
			fig = plt.figure()
			for ctr, curr_var_key in enumerate(var_keys):
				## Plot for result for initial entry points in graph
				# Dict mapping cost_bucket to dict containing {cost:<>, "recall":<>, "key":<actual_hyper_params_of_search>
				graph_res_dict = processed_data[fixed_key][curr_var_key][topk]["data"]["best"]
				X_bkts = sorted([_str_to_tuple(x, int) for x in graph_res_dict.keys()])
			
				# X = np.array([x[0] for x in X_bkts])
				X = [graph_res_dict[str(x)]["cost"] for x in X_bkts] # Use actual cost instead of cost bucket
				Y = [graph_res_dict[str(x)]["recall"] for x in X_bkts]
				
				
				
				## Plot for best result wrt graph search
				# Dict mapping cost_bucket to dict containing {cost:<>, "recall":<>, "key":<actual_hyper_params_of_search>
				init_res_dict = processed_data[fixed_key][curr_var_key][topk]["data"]["init"]
				X_bkts = sorted([_str_to_tuple(x, int) for x in init_res_dict.keys()])
			
				# X_init = np.array([x[0] for x in X_bkts])
				X_init = [init_res_dict[str(x)]["cost"] for x in X_bkts] # Use actual cost instead of cost bucket
				Y_init = [init_res_dict[str(x)]["recall"] for x in X_bkts]
				
				if plot_init_only:
					plt.plot(X_init, Y_init, "^--", c=cmap[ctr%len(cmap)][0], label=curr_var_key)
					LOGGER.info(f"X_init, Y_init {list(zip(X_init, Y_init))}")
				else:
					plt.plot(X, Y, "*-", label=curr_var_key, c=cmap[ctr%len(cmap)][0])
					plt.plot(X_init, Y_init, "^--", c=cmap[ctr%len(cmap)][1])
				
				
			# plt.ylim(0, 1)
			# plt.xlim(10, 2000)
			plt.grid()
			plt.legend(fontsize=6)
			plt.xlabel("Cost buckets")
			plt.ylabel("Recall")
			plt.title("\n".join(wrap(f"Fixed Key = {fixed_key}")))
			plt.xscale("log")
			fig.tight_layout()
			
			plt_fname = f"{base_plt_res_dir}/topk={topk}/{fixed_key}.pdf"
			Path(os.path.dirname(plt_fname)).mkdir(exist_ok=True, parents=True)
			plt.savefig(plt_fname)
			
			plt.close()


def main():
	
	# data_name_vals = ["pro_wrestling", "lego", "doctor_who"]
	# data_name_vals = ["pro_wrestling", "lego"]
	# data_name_vals = ["pro_wrestling", "lego", "yugioh"]
	# data_name_vals = ["pro_wrestling", "doctor_who"]
	data_name_vals = ["pro_wrestling"]
	# data_name_vals = ["doctor_who"]
	data_name_vals = ["lego"]
	data_name_vals = ["star_trek"]
	data_name_vals = ["yugioh", "star_trek", "lego", "pro_wrestling"]
	data_name_vals = ["yugioh"]
	
	

	for data_name in data_name_vals:
		process_results(data_name=data_name)
	pass

	
if __name__ == "__main__":
	main()
	
	

