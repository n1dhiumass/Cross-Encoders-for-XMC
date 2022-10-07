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


def process_res_file(file):
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
		
		processed_data = {}
		
		k_retvr_vals = [10, 50, 100, 200, 500, 1000]
		# k_retvr_vals = [200, 500]
		# anc_n_m_vals = [500]
		anc_n_m_vals = [50, 500]
		# anc_n_e_vals = [200, 800]
		anc_n_e_vals = [200]
		top_k_vals = [1, 100]
		
		for anc_n_m, anc_n_e, top_k in itertools.product(anc_n_m_vals, anc_n_e_vals, top_k_vals):
			
			try:
				processed_data[f"anc_n_m={anc_n_m}_anc_n_e={anc_n_e}_top_k={top_k}"] = {
					k_retvr: data["i_cur"][f"top_k={top_k}"][f"k_retvr={k_retvr}"][f"anc_n_m={anc_n_m}~anc_n_e={anc_n_e}"]["non_anchor"]
					for k_retvr in k_retvr_vals if k_retvr > top_k
				}
			except KeyError as e:
				# raise e
				pass
				

		return processed_data
	except Exception as e:
		# embed()
		# input("")
		raise e



def process_res_for_rq(data, data_name, fixed_params, var_params, updated_all_param_vals={}):
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

	template = "i_cur_n_steps={i_cur_n_steps}_sampling_method={sampling_method}_shortlist_method={shortlist_method}_anc_n_m={anc_n_m}_anc_n_e={anc_n_e}_top_k={top_k}"
	
	
	all_param_vals = {

		# "shortlist_method": ["exact", "approx"],
		"shortlist_method": ["none"],
		
		# "sampling_method": ["random_cumul", "approx_cumul", "approx_softmax_cumul", "exact_cumul", "exact_softmax_cumul",
		# 					"random_diff", "approx_diff", "approx_softmax_diff", "exact_diff", "exact_softmax_diff"],
		"sampling_method": ["bienc", "random_cumul",
							"approx_softmax_cumul", "approx_topk_cumul",
							# "variance_cumul", "variance_topk_cumul",
							"exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul"],
		# "sampling_method": ["random_cumul", "approx_softmax_cumul", "exact_softmax_cumul"],
		
		"i_cur_n_steps": [1, 2, 5, 10, 20, 50, 100, 200],
		# "i_cur_n_steps": [1, 2, 5],
		
		
		"anc_n_m": [50, 500],
		# "anc_n_e": [200, 800],
		"anc_n_e": [200],
		"top_k": [1, 100],
	}
	all_param_vals.update(updated_all_param_vals)
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
			
			# embed()
			# input("")
			if comb_key not in data: continue
			final_res[fixed_key][var_key] = data[comb_key]
		
	
	return final_res


def create_combined_result_file(base_res_dir):
	
	
	file_list = glob.glob(f"{base_res_dir}/*/retrieval_wrt_exact_crossenc.json")

	LOGGER.info(f"Processing {len(file_list)} files")
	all_data = {}
	for full_filename in tqdm(sorted(file_list)):
		rel_filename = full_filename.split("/")[-2]
		# rel_filename = rel_filename[:-5] if rel_filename.endswith(".json") else rel_filename

		try:
			curr_data = process_res_file(file= full_filename)
		except Exception as e:
			LOGGER.info(f"Error processing file {rel_filename}")
			continue

		if len(curr_data) == 0:
			LOGGER.info(f"No data in file {rel_filename}")
			continue

		all_data.update({rel_filename+"_"+key:val for key,val in curr_data.items()})
		
		if rel_filename == "i_cur_n_steps=1_sampling_method=bienc_shortlist_method=none":
			for step in [1, 2,5, 10, 20, 50]:
				temp_rel_filename = f"i_cur_n_steps={step}_sampling_method=bienc_shortlist_method=none"
				all_data.update({temp_rel_filename+"_"+key:val for key,val in curr_data.items()})
		
	LOGGER.info(f"Data successfully read for {len(all_data)} files")
	
	# Saving intermediate data
	inter_fname = f"{base_res_dir}/RQ/inter_res.json"
	Path(os.path.dirname(inter_fname)).mkdir(exist_ok=True, parents=True)
	with open(inter_fname, "w") as fout:
		json.dump(all_data, fout, indent=4)
	
	return inter_fname
	

def process_results(data_name):
	try:
		# CUR w/ Cross-Encoder scores
		base_res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/" \
					   f"score_mats_model-2-15999.0--79.46.ckpt/{data_name}/Retrieval_wrt_Exact_CrossEnc/"
		# base_res_dir += f"nm=100_ne={NUM_ENTS[data_name]}_s=5_i_cur_uniform_anchor_split"
		# base_res_dir += f"nm=100_ne={NUM_ENTS[data_name]}_s=5_i_cur_uniform_anchor_split_wo_shortlisting"
		# base_res_dir += f"nm=550_ne={NUM_ENTS[data_name]}_s=1_i_cur_uniform_anchor_split_wo_shortlisting_550_ments_anc_nm_500"
		# base_res_dir += f"nm=550_ne={NUM_ENTS[data_name]}_s=5_i_cur_uniform_anchor_split_wo_shortlisting"
		base_res_dir += f"nm=550_ne={NUM_ENTS[data_name]}_s=1_i_cur_uniform_anchor_split_wo_shortlisting"
		# base_res_dir += f"nm=100_ne={NUM_ENTS[data_name]}_s=5_i_cur_uniform_anchor_split_wo_shortlisting"
		
		# # CUR w/ Bi-Encoder scores
		# base_res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/" \
		# 			   f"score_mats_model-3-12039.0-2.17.ckpt/{data_name}/Retrieval_wrt_Exact_CrossEnc/"
		# # base_res_dir += f"nm=100_ne=10031_s=5_i_cur_uniform_anchor_split_wo_shortlisting_bienc"
		# base_res_dir += f"nm=550_ne=10031_s=5_i_cur_uniform_anchor_split_wo_shortlisting_bienc_anc_nm_500"
		
		# base_res_dir = f"../../results/6_ReprCrossEnc/Graph_Search/{data_name}/node_masking/{data_name}" # FIXME: Use for debuggging only
		
		# LOGGER.info("Skipping creating intermediate file, using existing file")
		inter_fname = create_combined_result_file(base_res_dir=base_res_dir)
		
		# Loading intermediate data
		LOGGER.info(f"Processing data for each research question for {data_name}")
		with open(inter_fname, "r") as fin:
			all_data = json.load(fin)
		
		
		RQs = {
			"RQ/1_Effect_of_sampling_method_on_CUR": {
				"fixed_params":["top_k", "i_cur_n_steps", "shortlist_method", "anc_n_m", "anc_n_e"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
				}
			},
			"RQ/2_Effect_of_num_i_cur_steps": {
				"fixed_params":["top_k", "sampling_method", "shortlist_method", "anc_n_m", "anc_n_e"],
				"var_params":["i_cur_n_steps"],
				"all_param_vals": {
					"shortlist_method": ["none"],
				}
			},
			# "RQ/3_Effect_of_shortlisting_method": {
			# 	"fixed_params":["top_k", "sampling_method", "i_cur_n_steps", "anc_n_m", "anc_n_e"],
			# 	"var_params":["shortlist_method"],
			# 	"all_param_vals": {}
			# },
			# "RQ/4_Best_Possible_Performance_Using_Exact_Scores": {
			# 	"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
			# 	"var_params":["sampling_method"],
			# 	"all_param_vals": {
			# 		"shortlist_method": ["exact"],
			# 		"sampling_method": ["random_cumul", "exact_cumul", "exact_softmax_cumul"],
			# 	},
			# 	"metric": "exact_vs_reranked_approx_retvr~common_frac_mean"
			# },
			# "RQ/4_Best_Possible_Performance_Using_Exact_Scores_error/gt_in_anc": {
			# 	"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
			# 	"var_params":["sampling_method"],
			# 	"all_param_vals": {
			# 		"shortlist_method": ["exact"],
			# 		"sampling_method": ["random_cumul", "exact_cumul", "exact_softmax_cumul"],
			# 	},
			# 	"metric": "exact_vs_anchor_ents~common_frac_mean"
			# },
			"RQ/5_Actual_Performance_Using_Approx_Scores": {
				"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
					"sampling_method": ["random_cumul", "exact_softmax_cumul",
										"approx_cumul", "approx_softmax_cumul", "approx_topk_cumul",
										# "variance_cumul", "variance_topk_cumul"
										],
				},
				"metric": "exact_vs_reranked_approx_retvr~common_frac_mean"
			},
			"RQ/6_Oracle_Performance_Using_Exact_Scores/recall": {
				"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
					"sampling_method": ["random_cumul", "exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul"],
				},
				"metric": "exact_vs_reranked_approx_retvr~common_frac_mean"
			},
			"RQ/6_Oracle_Performance_Using_Exact_Scores/gt_in_anc": {
				"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
					"sampling_method": ["random_cumul", "exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul"],
				},
				"metric": "exact_vs_anchor_ents~common_frac_mean"
			},
		}
		
		percent_error_RQs = {
			f"RQ/6_Oracle_Performance_Using_Exact_Scores/percent_error/{ent_type}": {
				"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
					"sampling_method": ["random_cumul", "approx_softmax_cumul",
										"exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul"],
				},
				"metric": f"approx_error_{ent_type}_percent"
			} for ent_type in ["head", "all", "anc_ents"]
			
		}
		
		l1_error_RQs = {
			f"RQ/6_Oracle_Performance_Using_Exact_Scores/l1_error/{ent_type}": {
				"fixed_params":["top_k", "anc_n_m", "anc_n_e", "shortlist_method", "i_cur_n_steps"],
				"var_params":["sampling_method"],
				"all_param_vals": {
					"shortlist_method": ["none"],
					"sampling_method": ["random_cumul", "approx_softmax_cumul",
										"exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul"],
				},
				"metric": f"approx_error_{ent_type}"
			}
			for ent_type in ["head", "all", "anc_ents"]
		}
		
		RQs.update(percent_error_RQs)
		RQs.update(l1_error_RQs)
		
		for curr_rq in RQs:
			LOGGER.info(f"Processing data for RQ : {curr_rq}")
			processed_res = process_res_for_rq(
				data_name=data_name,
				data=all_data,
				fixed_params=RQs[curr_rq]["fixed_params"],
				var_params=RQs[curr_rq]["var_params"],
				updated_all_param_vals=RQs[curr_rq]["all_param_vals"],
			)
			# Save data
			process_fname = f"{base_res_dir}/{curr_rq}/processed_res.json"
			Path(os.path.dirname(process_fname)).mkdir(exist_ok=True, parents=True)
			with open(process_fname, "w") as fout:
				json.dump(processed_res, fout, indent=4)
			
			# Plot data
			plot_processed_results(
				res_fname=process_fname,
				metric=RQs[curr_rq]["metric"] if "metric" in RQs[curr_rq] else "exact_vs_reranked_approx_retvr~common_mean"
			)

			
	except Exception as e:
		embed()
		raise e
	
	
def plot_processed_results(res_fname, metric):
	

	base_plt_res_dir = f"{os.path.dirname(res_fname)}/plots"
	Path(base_plt_res_dir).mkdir(exist_ok=True, parents=True)
	with open(res_fname, "r") as fin:
		processed_data = json.load(fin)
	
	style_map = {
		"sampling_method=bienc": {"color":"firebrick", "marker":"*", "linestyle":"dashed"},
		"sampling_method=random_cumul": {"color":"darkorange", "marker":"*", "linestyle":"dashed"},
		"sampling_method=approx_topk_cumul": {"color":"green", "marker":"*", "linestyle":"solid"},
		"sampling_method=approx_softmax_cumul": {"color":"olive", "marker":"s", "linestyle":"solid"},
		"sampling_method=exact_topk_cumul": {"color":"navy", "marker":"*", "linestyle":"solid"},
		"sampling_method=exact_softmax_cumul": {"color":"turquoise", "marker":"s", "linestyle":"solid"},
		"sampling_method=exact_after_topk_cumul": {"color":"deepskyblue", "marker":"v", "linestyle":"solid"},
	}

	for fixed_key in processed_data:
		var_keys = processed_data[fixed_key].keys()
		
		plt.clf()
		fig = plt.figure()
		for ctr, curr_var_key in enumerate(var_keys):
			## Plot for result for initial entry points in graph
			# Dict mapping cost to dict containing all metrics
			curr_res_dict = processed_data[fixed_key][curr_var_key]
			X = np.array(sorted([int(x) for x in curr_res_dict.keys()]))
		
			Y = [curr_res_dict[str(x)][metric] for x in X]
			
			# To adjust for the fact that CUR uses additional cross-encoder calls for embedddings the query
			# if "bienc" not in curr_var_key:
			# 	X += 200
			m = style_map[curr_var_key]["marker"] if curr_var_key in style_map else "*"
			lstyle = style_map[curr_var_key]["linestyle"] if curr_var_key in style_map else "solid"
			c = style_map[curr_var_key]["color"] if curr_var_key in style_map else cmap[ctr%len(cmap)][0]
			if "i_cur" in curr_var_key:
				# c = plt.cm.Greens(np.linspace(0, 1, 20))[10 + ctr%10]
				c = plt.cm.winter(np.linspace(0, 1, len(var_keys)))[ctr%len(var_keys)]
				
			plt.plot(X, Y, linestyle=lstyle, marker=m, label=curr_var_key, c=c)
			
		plt.grid()
		plt.legend(fontsize=6)
		plt.xlabel("Number of Items Retrieved")
		plt.ylabel(metric)
		plt.title("\n".join(wrap(f"Fixed Key = {fixed_key}", width=50)))
		# plt.title("\n".join(wrap(f"Fixed Key = {'~'.join(fixed_key.split('='))}")))
		plt.xscale("log")
		# fig.tight_layout()
		
		plt_fname = f"{base_plt_res_dir}/{fixed_key}.pdf"
		Path(os.path.dirname(plt_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(plt_fname)
		
		plt.close()


def main():
	
	data_name_vals = ["yugioh", "star_trek", "lego", "pro_wrestling"]
	# data_name_vals = ["yugioh"]
	data_name_vals = ["lego", "star_trek"]
	data_name_vals = ["star_trek"]
	
	

	for data_name in data_name_vals:
		LOGGER.info(f"Processing results for {data_name}")
		process_results(data_name=data_name)
	pass

	
if __name__ == "__main__":
	main()
	
	

