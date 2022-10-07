import os
import sys
import json
import torch
import pickle
import logging
import argparse
import wandb
import warnings
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse.linalg import svds
from scipy.linalg import svd, diagsvd
from sklearn.decomposition import NMF

from eval.eval_utils import compute_label_embeddings, compute_input_embeddings, compute_overlap, compute_recall
from eval.matrix_approx_zeshel import CURApprox, plot_heat_map
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS
from eval.test_svd_for_matrix_factorization import sparsify_mat

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def _get_indices_scores(topk_preds):
	"""
	Convert a list of indices,scores tuple to two list by concatenating all indices and all scores together.
	:param topk_preds: List of indices,scores tuple
	:return: dict with two keys "indices" and "scores" mapping to lists
	"""
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy(), scores.cpu().numpy()
	return {"indices":indices, "scores":scores}


def get_error(exact_mat, approx_mat, cols_for_each_row):
	
	
	assert exact_mat.shape == approx_mat.shape
	assert exact_mat.shape[0] == len(cols_for_each_row)
	assert len(cols_for_each_row.shape) == 2
	
	diff =  approx_mat - exact_mat
	diff_for_cols = torch.stack([diff[row_idx, cols_for_curr_row] for row_idx, cols_for_curr_row in enumerate(cols_for_each_row)])
	exact_for_cols = torch.stack([exact_mat[row_idx, cols_for_curr_row] for row_idx, cols_for_curr_row in enumerate(cols_for_each_row)])
	
	error =  torch.norm(diff_for_cols, p=1).data.numpy()
	exact_mat_norm = torch.norm(exact_for_cols, p=1).data.numpy()
	rel_error = error/(cols_for_each_row.shape[0]*cols_for_each_row.shape[1])
	
	percent_error = torch.mean(torch.divide(diff_for_cols, exact_for_cols))
	percent_error = percent_error.cpu().numpy()
	
	abs_percent_error = torch.mean(torch.abs(torch.divide(diff_for_cols, exact_for_cols)))
	abs_percent_error = abs_percent_error.cpu().numpy()
	
	return error, rel_error, percent_error, abs_percent_error

	
def _get_variance_probs(curr_anchor_ents, curr_ment_to_ent_scores, anchor_rows, subset_size, num_subsets, entities_for_anchoring):
	"""
	Get probability distribution for sampling entities based on uncertainty in approximate scores
	:param curr_anchor_ents:
	:param curr_ment_to_ent_scores:
	:param anchor_rows:
	:param subset_size:
	:param num_subsets:
	:return:
	"""

	try:

		rng = np.random.default_rng(seed=0)
		approx_ment_to_ent_scores = []
		if len(curr_anchor_ents) == 0 or subset_size == 0 or num_subsets == 0:
			return None
		
		for i in range(num_subsets):
			col_idxs = rng.choice(curr_anchor_ents, size=subset_size, replace=False)
		
			curr_cols = curr_ment_to_ent_scores[col_idxs]
		
			intersect_mat = anchor_rows[:, col_idxs] # kr x kc
		
			U = torch.tensor(np.linalg.pinv(intersect_mat)) # kc x kr
		
			approx_ment_to_ent_scores += [curr_cols @ U @ anchor_rows] # shape: (1, n_ents)
		
		approx_ment_to_ent_scores = torch.stack(approx_ment_to_ent_scores).cpu().numpy()
		approx_ment_to_ent_scores_mean = np.mean(approx_ment_to_ent_scores, axis=0)
		approx_ment_to_ent_scores_var = np.std(approx_ment_to_ent_scores, axis=0)
		
		
		
		prob_for_ents_for_anchoring = approx_ment_to_ent_scores_var[entities_for_anchoring]
		
		if np.sum(prob_for_ents_for_anchoring) > 0:
			prob_for_ents_for_anchoring = prob_for_ents_for_anchoring/np.sum(prob_for_ents_for_anchoring)
		else:
			prob_for_ents_for_anchoring += 1/len(entities_for_anchoring)
			
		# LOGGER.info("Intentional embed")
		# LOGGER.info(f"{approx_ment_to_ent_scores.shape}")
		# LOGGER.info(f"{approx_ment_to_ent_scores_mean.shape}")
		# LOGGER.info(f"{approx_ment_to_ent_scores_var.shape}")
		# LOGGER.info(f"{prob_for_ents_for_anchoring.shape}")
		# LOGGER.info(f"Probs  = {np.sum(prob_for_ents_for_anchoring)}")
		# embed()
		return prob_for_ents_for_anchoring
	except Exception as e:
		embed()
		raise e
	
	
	
def run_approx_eval_i_cur_w_seed(approx_method, all_ment_to_ent_scores, n_ment_anchors, n_ent_anchors, top_k, top_k_retvr, seed, icur_args):
	"""
	Takes a ground-truth mention x entity matrix as input, uses some rows and columns of this matrix to INCREMENTALLY approximate the entire matrix
	and also evaluate the approximation
	:param all_ment_to_ent_scores:
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param top_k_retvr:
	:param approx_method
	:param seed:
	:return:
	"""

	
	try:
		assert approx_method == "i_cur"
		
		n_ments = all_ment_to_ent_scores.shape[0]
		n_ents = all_ment_to_ent_scores.shape[1]
		all_ent_set = set(list(range(n_ents)))
		rng = np.random.default_rng(seed=seed)
	
		
		shortlist_method = icur_args["shortlist_method"]
		sampling_method = icur_args["sampling_method"]
		i_cur_n_steps = icur_args["i_cur_n_steps"]
		res_dir = icur_args["res_dir"]
		assert i_cur_n_steps > 0
		assert shortlist_method == "none", f"shortlist_method = {shortlist_method} not supported"
		
		n_ent_anchors_per_step_vals = [int(n_ent_anchors/i_cur_n_steps)]*(i_cur_n_steps-1)
		n_ent_anchors_per_step_vals += [n_ent_anchors - sum(n_ent_anchors_per_step_vals)]
		
		# # Define number of entities shortlisted iteratively in each step
		# top_k_retvr_per_step_vals = sorted(np.arange(top_k_retvr, n_ents, (n_ents-top_k_retvr)/i_cur_n_steps), reverse=True)
		# top_k_retvr_per_step_vals = [int(x) for x in top_k_retvr_per_step_vals]
		
		assert sum(n_ent_anchors_per_step_vals) == n_ent_anchors, f"sum(n_ent_anchors_per_step_vals) = {sum(n_ent_anchors_per_step_vals)} != n_ent_anchors = {n_ent_anchors}"
		
		anchor_ment_idxs = sorted(rng.choice(n_ments, size=n_ment_anchors, replace=False))
		row_idxs = anchor_ment_idxs
		anchor_rows = all_ment_to_ent_scores[row_idxs,:]
		
		non_anchor_ment_idxs = sorted(list(set(list(range(n_ments))) - set(anchor_ment_idxs)))
		
		# Initially, all anchor entities can be chosen from all entities
		entities_for_anchoring = [np.arange(n_ents) for _ in range(n_ments)]
		approx_ment_to_ent_scores = torch.zeros((n_ments, n_ents))
		
		LOGGER.info(f"n_ent_anchors_per_step_vals {n_ent_anchors_per_step_vals}")
		# LOGGER.info(f"top_k_retvr_per_step_vals {top_k_retvr_per_step_vals}")
		anchor_ent_idxs_per_step = [[] for ment_idx in range(n_ments)]
		for step in range(i_cur_n_steps):
			# LOGGER.info(f"Step {step}")
			curr_n_ent_anchors = n_ent_anchors_per_step_vals[step]
			# curr_top_k_retvr = top_k_retvr_per_step_vals[step]
			# for ment_idx in non_anchor_ment_idxs:
			for ment_idx in range(n_ments):
				
				
				if sampling_method == "random_cumul":
					# Sampling uniformly at random, and using cumulative set of anchor entities
					anchor_ent_idxs_per_step[ment_idx] += sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False))
				elif sampling_method == "random_diff":
					# Sampling uniformly at random, and different set of anchor entities for each step
					anchor_ent_idxs_per_step[ment_idx] = sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False))
				elif sampling_method in ["approx_cumul", "approx_diff", "approx_softmax_cumul", "approx_softmax_diff", "approx_topk_cumul"]:
					# Sampling based on approx scores
					sample_prob = approx_ment_to_ent_scores[ment_idx][entities_for_anchoring[ment_idx]]
					a = approx_ment_to_ent_scores[ment_idx][entities_for_anchoring[ment_idx]]
					if sampling_method in ["approx_softmax_cumul", "approx_softmax_diff"]:
						softmax = torch.nn.Softmax(dim=0)
						sample_prob = softmax(sample_prob) + 1e-20 # Adding some smoothening so that we don't have less than curr_n_ent_anchors non-zero entries
						
					if torch.sum(sample_prob) == 0:
						sample_prob = sample_prob + 1/len(entities_for_anchoring[ment_idx])
					else:
						sample_prob = sample_prob/torch.norm(sample_prob, p=1)
					sample_prob = sample_prob.cpu().numpy()
					
					if sampling_method in ["approx_cumul", "approx_softmax_cumul"]:
						# and using cumulative set of anchor entities
						anchor_ent_idxs_per_step[ment_idx] += sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False, p=sample_prob))
					elif sampling_method in ["approx_topk_cumul"]:
						_, temp_top_prob_idxs = torch.topk(torch.tensor(sample_prob), k=curr_n_ent_anchors)
						temp_top_prob_idxs = temp_top_prob_idxs.cpu().numpy()
						anchor_ent_idxs_per_step[ment_idx] += entities_for_anchoring[ment_idx][temp_top_prob_idxs].tolist()
					elif sampling_method in ["approx_diff", "approx_softmax_diff"]:
						# or use different set of anchor entities for each step
						anchor_ent_idxs_per_step[ment_idx] = sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False, p=sample_prob))
					else:
						raise NotImplementedError(f"sampling_method = {sampling_method} not supported")
						
				elif sampling_method in ["exact_cumul", "exact_diff", "exact_softmax_cumul", "exact_softmax_diff", "exact_topk_cumul", "exact_after_topk_cumul"]:
					# Sampling based on exact cross-enc scores
					sample_prob = all_ment_to_ent_scores[ment_idx][entities_for_anchoring[ment_idx]]
					if sampling_method in ["exact_softmax_cumul", "exact_softmax_diff"]:
						softmax = torch.nn.Softmax(dim=0)
						sample_prob = softmax(sample_prob)
						
					if torch.sum(sample_prob) == 0:
						sample_prob = sample_prob + 1/len(entities_for_anchoring[ment_idx])
					else:
						sample_prob = sample_prob/torch.norm(sample_prob, p=1)
					sample_prob = sample_prob.cpu().numpy()
					
					if sampling_method in ["exact_cumul", "exact_softmax_cumul"]:
						# and using cumulative set of anchor entities
						anchor_ent_idxs_per_step[ment_idx] += sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False, p=sample_prob))
					elif sampling_method in ["exact_topk_cumul"]:
						_, temp_top_prob_idxs = torch.topk(torch.tensor(sample_prob), k=curr_n_ent_anchors)
						temp_top_prob_idxs = temp_top_prob_idxs.cpu().numpy()
						anchor_ent_idxs_per_step[ment_idx] += entities_for_anchoring[ment_idx][temp_top_prob_idxs].tolist()
					elif sampling_method in ["exact_after_topk_cumul"]:
						# Use top-entities other than top_k entities
						_, temp_top_prob_idxs = torch.topk(torch.tensor(sample_prob), k=curr_n_ent_anchors+top_k)
						temp_top_prob_idxs = temp_top_prob_idxs.cpu().numpy()
						temp_top_prob_idxs = temp_top_prob_idxs[top_k:] # Remove top_k idxs
						anchor_ent_idxs_per_step[ment_idx] += entities_for_anchoring[ment_idx][temp_top_prob_idxs].tolist()
					
					elif sampling_method == ["exact_diff", "exact_softmax_diff"]:
						# or use different set of anchor entities for each step
						anchor_ent_idxs_per_step[ment_idx] = sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False, p=sample_prob))
					else:
						raise NotImplementedError(f"sampling_method = {sampling_method} not supported")
				
				elif sampling_method in ["variance_cumul", "variance_topk_cumul"]:
					
					# sample_prob = all_ment_to_ent_scores[ment_idx][entities_for_anchoring[ment_idx]]
					sample_prob = _get_variance_probs(
						anchor_rows=anchor_rows,
						curr_ment_to_ent_scores=all_ment_to_ent_scores[ment_idx],
						curr_anchor_ents=anchor_ent_idxs_per_step[ment_idx],
						subset_size=int(len(anchor_ent_idxs_per_step[ment_idx])/2),
						num_subsets=10,
						entities_for_anchoring=entities_for_anchoring[ment_idx]
					)
					if sampling_method == "variance_cumul":
						anchor_ent_idxs_per_step[ment_idx] += sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False, p=sample_prob))
					elif sampling_method == "variance_topk_cumul":
						if sample_prob is None: # Choose randomly
							anchor_ent_idxs_per_step[ment_idx] += sorted(rng.choice(entities_for_anchoring[ment_idx], size=curr_n_ent_anchors, replace=False))
						else: # Pick top curr_n_ent_anchors entities
							_, temp_top_prob_idxs = torch.topk(torch.tensor(sample_prob), k=curr_n_ent_anchors)
							temp_top_prob_idxs = temp_top_prob_idxs.cpu().numpy()
							anchor_ent_idxs_per_step[ment_idx] += entities_for_anchoring[ment_idx][temp_top_prob_idxs].tolist()
					else:
						raise NotImplementedError(f"sampling_method = {sampling_method} not supported")
				else:
					raise NotImplementedError(f"sampling_method = {sampling_method} not supported")
					
				anchor_ent_idxs = anchor_ent_idxs_per_step[ment_idx]
				
				col_idxs = anchor_ent_idxs
				curr_cols = all_ment_to_ent_scores[ment_idx, col_idxs]
				
				intersect_mat = anchor_rows[:, col_idxs] # kr x kc
				
				U = torch.tensor(np.linalg.pinv(intersect_mat)) # kc x kr
				# # Oracle CUR approximation that uses entire test-time matrix to compute its low rank approximation
				# temp_A = torch.concat((anchor_rows, all_ment_to_ent_scores[ment_idx].unsqueeze(0)))
				# temp_C = temp_A[:, col_idxs]
				# U = torch.tensor(np.linalg.pinv(temp_C)) @ torch.tensor(temp_A) @ torch.tensor(np.linalg.pinv(anchor_rows))
				
				approx_ment_to_ent_scores[ment_idx] = curr_cols @ U @ anchor_rows # shape: (1, n_ents)
				
				# if shortlist_method in ["approx", "exact"]:
				# 	masked_entities = sorted(list(all_ent_set - set(entities_for_anchoring[ment_idx])))
				# 	approx_ment_to_ent_scores[ment_idx][masked_entities] = -99999999999999999
				#
				# 	if shortlist_method == "approx":
				# 		# Using approx scores for shortlisting top-k entities
				# 		_, temp_top_k_indices = approx_ment_to_ent_scores[ment_idx].topk(curr_top_k_retvr)
				# 	elif shortlist_method == "exact":
				# 		# Using exact scores for shortlisting top-k entities
				# 		_, temp_top_k_indices = all_ment_to_ent_scores[ment_idx].topk(curr_top_k_retvr)
				# 	else:
				# 		raise NotImplementedError(f"shortlist_method = {shortlist_method} not supported")
				#
				# 	entities_for_anchoring[ment_idx] = temp_top_k_indices.cpu().numpy().astype(int)
				# elif shortlist_method == "none":
				# 	pass
				# else:
				# 	raise NotImplementedError(f"shortlist_method = {shortlist_method} not supported")
				
				assert shortlist_method == "none", f"shortlist_method = {shortlist_method} not supported"
				
				# Remove already sampled anchor entities from allowed list of entities for anchoring
				entities_for_anchoring[ment_idx] = np.array(sorted(list(set(entities_for_anchoring[ment_idx]) - set(anchor_ent_idxs_per_step[ment_idx]))))
			
			# Break statement to evaluate recall after first round of shortlisting
			# top_k_retvr = curr_top_k_retvr
			# break
		
		
		topk_preds = []
		approx_topk_preds = []
		topk_w_approx_retrvr_preds = []
		
	
		for ment_idx in range(n_ments):
	
			curr_ment_scores = all_ment_to_ent_scores[ment_idx]
			approx_curr_ment_scores = approx_ment_to_ent_scores[ment_idx]
	
			# Get top-k indices from exact matrix
			top_k_scores, top_k_indices = curr_ment_scores.topk(top_k)
	
			# Get top-k indices from approx-matrix
			approx_top_k_scores, approx_top_k_indices = approx_curr_ment_scores.topk(top_k_retvr)
			
			# Re-rank top-k indices from approx-matrix using exact scores from ment_to_ent matrix
			# Scores from given ment_to_ent matrix filled only for entities retrieved by approximate retriever
			temp = torch.zeros(curr_ment_scores.shape) - 99999999999999
			temp[approx_top_k_indices] = curr_ment_scores[approx_top_k_indices]
		
			top_k_w_approx_retrvr_scores, top_k_w_approx_retrvr_indices = temp.topk(top_k)
			
			topk_preds += [(top_k_indices.unsqueeze(0), top_k_scores.unsqueeze(0))]
			approx_topk_preds += [(approx_top_k_indices.unsqueeze(0), approx_top_k_scores.unsqueeze(0))]
			topk_w_approx_retrvr_preds += [(top_k_w_approx_retrvr_indices.unsqueeze(0), top_k_w_approx_retrvr_scores.unsqueeze(0))]
	
		
		topk_preds = _get_indices_scores(topk_preds)
		approx_topk_preds = _get_indices_scores(approx_topk_preds)
		topk_w_approx_retrvr_preds = _get_indices_scores(topk_w_approx_retrvr_preds)
		
		def score_topk_preds_wrapper(arg_ment_idxs):
			# exact_vs_approx_retvr = compute_overlap(
			# 	indices_list1=topk_preds["indices"][arg_ment_idxs],
			# 	indices_list2=approx_topk_preds["indices"][arg_ment_idxs],
			# )
			# new_exact_vs_approx_retvr = {}
			# for _metric in exact_vs_approx_retvr:
			# 	new_exact_vs_approx_retvr[f"{_metric}_mean"] = float(exact_vs_approx_retvr[_metric][0][5:])
			# 	new_exact_vs_approx_retvr[f"{_metric}_std"] = float(exact_vs_approx_retvr[_metric][1][4:])
			# 	new_exact_vs_approx_retvr[f"{_metric}_p50"] = float(exact_vs_approx_retvr[_metric][2][4:])
			
			exact_vs_reranked_approx_retvr = compute_overlap(
				indices_list1=topk_preds["indices"][arg_ment_idxs],
				indices_list2=topk_w_approx_retrvr_preds["indices"][arg_ment_idxs],
			)
			new_exact_vs_reranked_approx_retvr = {}
			for _metric in exact_vs_reranked_approx_retvr:
				new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
			
			# Compare overlap between anchor entities and exact top-k entities
			exact_vs_anchor_ents = compute_recall(
				gt_indices_list=topk_preds["indices"][arg_ment_idxs],
				pred_indices_list=np.array(anchor_ent_idxs_per_step)[arg_ment_idxs],
			)
			
			new_exact_vs_anchor_ents = {}
			for _metric in exact_vs_anchor_ents:
				new_exact_vs_anchor_ents[f"{_metric}_mean"] = float(exact_vs_anchor_ents[_metric][0][5:])
				new_exact_vs_anchor_ents[f"{_metric}_std"] = float(exact_vs_anchor_ents[_metric][1][4:])
				new_exact_vs_anchor_ents[f"{_metric}_p50"] = float(exact_vs_anchor_ents[_metric][2][4:])
		
			res = {
				# f"exact_vs_approx_retrvr": new_exact_vs_approx_retvr,
				f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr,
				f"exact_vs_anchor_ents" : new_exact_vs_anchor_ents,
			}
			
			new_res = {}
			for res_type in res:
				for metric in res[res_type]:
					new_res[f"{res_type}~{metric}"] = res[res_type][metric]
			
		
			# new_res["approx_error"] =  (torch.norm( (approx_ment_to_ent_scores - all_ment_to_ent_scores)[arg_ment_idxs, :] )).data.numpy()
			# new_res["approx_error_relative"] =  new_res["approx_error"]/(torch.norm(all_ment_to_ent_scores[arg_ment_idxs, :]).data.numpy())
			
			new_res["approx_error_all"], new_res["approx_error_all_relative"], \
			new_res["approx_error_all_percent"], new_res["approx_error_all_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=np.array([np.arange(n_ents) for _ in arg_ment_idxs])
			)
			
			_, gt_top_k_ent_idxs = torch.topk(all_ment_to_ent_scores[arg_ment_idxs], k=top_k)
			new_res["approx_error_head"], new_res["approx_error_head_relative"], \
			new_res["approx_error_head_percent"], new_res["approx_error_head_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=gt_top_k_ent_idxs
			)
			
			_, gt_tail_ent_idxs = torch.topk(-1*all_ment_to_ent_scores[arg_ment_idxs], k=n_ents - top_k)
			new_res["approx_error_tail"], new_res["approx_error_tail_relative"], \
			new_res["approx_error_tail_percent"], new_res["approx_error_tail_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=gt_tail_ent_idxs
			)
			
			new_res["approx_error_anc_ents"], new_res["approx_error_anc_ents_relative"], \
			new_res["approx_error_anc_ents_percent"], new_res["approx_error_anc_ents_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=np.array(anchor_ent_idxs_per_step)[arg_ment_idxs]
			)
			
			
			
			return new_res
		
		final_res = {
			"anchor":score_topk_preds_wrapper(arg_ment_idxs=anchor_ment_idxs),
			"non_anchor":score_topk_preds_wrapper(arg_ment_idxs=non_anchor_ment_idxs),
			"all":score_topk_preds_wrapper(arg_ment_idxs=list(range(n_ments)))
		}
		
		score_matrices = {
			"exact": np.array(all_ment_to_ent_scores[non_anchor_ment_idxs]),
			"approx": np.array(approx_ment_to_ent_scores[non_anchor_ment_idxs]),
		}
		
		if sampling_method == "exact_after_topk_cumul":
			out_dir = f"{res_dir}/score_plots_non_anchor_anc_nm={n_ment_anchors}_anc_ne={n_ent_anchors}_seed={seed}_topk={top_k}"
		else:
			out_dir = f"{res_dir}/score_plots_non_anchor_anc_nm={n_ment_anchors}_anc_ne={n_ent_anchors}_seed={seed}"
			
		plot_score_distribution(
			out_dir=out_dir,
			score_matrices=score_matrices,
			num_rows=len(non_anchor_ment_idxs),
			anchor_ent_idxs=np.array(anchor_ent_idxs_per_step)[non_anchor_ment_idxs]
		)
		
		return final_res
	except Exception as e:
		embed()
		raise e


def run_approx_eval_w_seed(approx_method, all_ment_to_ent_scores, n_ment_anchors, n_ent_anchors, top_k, top_k_retvr,
						   seed, precomp_approx_ment_to_ent_scores):
	"""
	Takes a ground-truth mention x entity matrix as input, uses some rows and columns of this matrix to approximate the entire matrix
	and also evaluate the approximation
	:param all_ment_to_ent_scores:
	:param precomp_approx_ment_to_ent_scores
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param top_k_retvr:
	:param approx_method
	:param seed:
	:return:
	"""

	
	try:
		n_ments = all_ment_to_ent_scores.shape[0]
		n_ents = all_ment_to_ent_scores.shape[1]
	
		rng = np.random.default_rng(seed=seed)
	
		anchor_ment_idxs = sorted(rng.choice(n_ments, size=n_ment_anchors, replace=False))
		anchor_ent_idxs = sorted(rng.choice(n_ents, size=n_ent_anchors, replace=False))
		row_idxs = anchor_ment_idxs
		col_idxs = anchor_ent_idxs
		rows = all_ment_to_ent_scores[row_idxs,:]
		cols = all_ment_to_ent_scores[:,col_idxs]
	
		non_anchor_ment_idxs = sorted(list(set(list(range(n_ments))) - set(anchor_ment_idxs)))
		non_anchor_ent_idxs = sorted(list(set(list(range(n_ents))) - set(anchor_ent_idxs)))
	
		if approx_method in ["bienc","fixed_anc_ent"] or approx_method.startswith("fixed_anc_ent_cur_"):
			approx_ment_to_ent_scores = precomp_approx_ment_to_ent_scores
		elif approx_method == "cur":
			approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows")
			approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
		# elif approx_method == "i_cur": # Incremental CUR
		# 	# TODO: Fix this: Integrate i_cur such that we don't have to return result here, instead we just get approx_score matrix and use code below to evaluate the approximation
		# 	return run_approx_eval_i_cur_w_seed(
		# 		approx_method=approx_method,
		# 		all_ment_to_ent_scores=all_ment_to_ent_scores,
		# 		n_ment_anchors=n_ment_anchors,
		# 		n_ent_anchors=n_ent_anchors,
		# 		top_k=top_k,
		# 		top_k_retvr=top_k_retvr,
		# 		seed=seed,
		# 		icur_args=icur_args
		# 	)
		elif approx_method == "cur_oracle":
			approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows", A=all_ment_to_ent_scores)
			approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
		elif approx_method == "svd_sparse":
			matrix_for_svd = np.zeros((n_ments, n_ents))
			matrix_for_svd[row_idxs] = rows
			# matrix_for_svd[non_anchor_ment_idxs, anchor_ent_idxs] = all_ment_to_ent_scores[non_anchor_ment_idxs, anchor_ent_idxs]
			for non_anc_iter in non_anchor_ment_idxs:
				matrix_for_svd[non_anc_iter, anchor_ent_idxs] = all_ment_to_ent_scores[non_anc_iter, anchor_ent_idxs]
			
			rank = n_ments-1
			U, S, VT = svds(matrix_for_svd, k=rank, solver="arpack")
			# U, S, VT = svds(matrix_for_svd, k=rank, solver="lobpcg")
			# U.shape = (n_ments, rank)
			approx_ment_to_ent_scores = torch.tensor(U @ np.diag(S) @ VT)
		elif approx_method == "cur_w_svd_sparse":
			
			# U, S, VT = svd(rows, full_matrices=False)
			
			matrix_for_svd = sparsify_mat(dense_mat=rows.cpu().numpy(), n_sparse_per_row=int(0.5*n_ents), seed=seed)
			
			model = NMF(init='nndsvd', n_components=n_ment_anchors-1, random_state=0)
			W = model.fit_transform(matrix_for_svd)
			approx_dense_mat = W @ model.components_
			# embed()
			# input("")
			
			# rank=n_ment_anchors-1
			# U, S, VT = svds(matrix_for_svd, k=rank, solver="arpack")
			# # U, S, VT = svds(matrix_for_svd, k=rank, solver="lobpcg")
			# approx_dense_mat = torch.tensor(U @ np.diag(S) @ VT)
			
			dummy_ment_to_ent_score_mat = np.zeros((n_ments, n_ents))
			dummy_ment_to_ent_score_mat[row_idxs] = approx_dense_mat
			for non_anc_iter in non_anchor_ment_idxs:
				dummy_ment_to_ent_score_mat[non_anc_iter, anchor_ent_idxs] = all_ment_to_ent_scores[non_anc_iter, anchor_ent_idxs]
			
			dummy_ment_to_ent_score_mat = torch.tensor(dummy_ment_to_ent_score_mat)
			approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs,
							   rows=dummy_ment_to_ent_score_mat[row_idxs, :],
							   cols=dummy_ment_to_ent_score_mat[:, col_idxs], approx_preference="rows")
			approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
		
		elif approx_method in ["cur_linreg", "svd_linreg", "svd_linreg_well_cond"]:
			if approx_method == "cur_linreg":
				approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows")
				VT = approx.latent_cols.cpu().numpy()
				V = VT.T
				# V.shape : (n_ents, K=n_anc_ents)
			elif approx_method == "svd_linreg":
				matrix_for_svd = rows
				U, S, VT = svd(matrix_for_svd, full_matrices=False)
				V = VT.T
				# U.shape : (n_anc_ments, K)
				# S.shape : (K,) where K = min(n_anc_ments, n_ents)
				# V.shape : (n_ents, K)
			elif approx_method == "svd_linreg_well_cond":
				matrix_for_svd = rows
				U, S, VT = svd(matrix_for_svd, full_matrices=False)
				
				K = min(n_ment_anchors, n_ent_anchors)
				# Taking on top-K singular vectors so that the linear regression problem has
				# number of variables <= # of equations. This is some sort of regularization that improve performance for the matrix completion task.
				U = U[:, :K]
				S = S[:K]
				VT = VT[:K]
				
				V = VT.T
				# U.shape : (n_anc_ments, K)
				# S.shape : (K,) where K = min(n_anc_ments, n_anc_ents)
				# V.shape : (n_ents, K)
			else:
				raise NotImplementedError(f"approx_method={approx_method}")
			
			approx_ment_to_ent_scores = np.zeros((n_ments, n_ents))
			approx_ment_to_ent_scores[row_idxs] = rows
			
			A = V[anchor_ent_idxs] # shape: (n_anc_ents, K)
			B = np.linalg.pinv(A.T @ A) @ A.T # shape :  K x n_anc_ents
			
			# if number of variables > number of equations
			# for CUR_linreg : number of variables = number of equations = n_anc_ents
			# for SVD_linreg : number of variables = min(n_anc_ments, n_ents) = n_anc_ments,  number of equations = n_anc_ents
			# for SVD_linreg_well_cond : number of variables = min(n_anc_ments, n_anc_ents),  number of equations = n_anc_ents
			if B.shape[0] > B.shape[1]:
				LOGGER.info(f"For method = {approx_method}, n_anc_ment={n_ment_anchors}, n_acn_ent={n_ent_anchors}, lin regression problem is underspecified")
				
			for non_anc_iter in non_anchor_ment_idxs:
				
				b = all_ment_to_ent_scores[non_anc_iter][anchor_ent_idxs].cpu().numpy() # Scores for anchor items
				
				# Solution to regression problem A x = b gives us query embedding
				# A: Item embeddings, x: query embedding, b: target query-item scores
				# B = np.linalg.pinv(A.T @ A) @ A.T 
				x = B @ b
				
				curr_approx_score_row = x @ VT # Compute approx scores using learnt query embedding
				approx_ment_to_ent_scores[non_anc_iter] = curr_approx_score_row
			
			approx_ment_to_ent_scores = torch.tensor(approx_ment_to_ent_scores)
		
		elif approx_method == "svd_oracle":
			matrix_for_svd = all_ment_to_ent_scores
			U, S, VT = svd(matrix_for_svd)
			# U.shape = (n_ments, rank)
			approx_ment_to_ent_scores = torch.tensor(U @ diagsvd(S, n_ments, n_ents) @ VT)
		else:
			raise NotImplementedError(f"approx_method = {approx_method} not supported")
		
		topk_preds = []
		approx_topk_preds = []
		topk_w_approx_retrvr_preds = []
		
	
		for ment_idx in range(n_ments):
	
			curr_ment_scores = all_ment_to_ent_scores[ment_idx]
			approx_curr_ment_scores = approx_ment_to_ent_scores[ment_idx]
	
			# Get top-k indices from exact matrix
			top_k_scores, top_k_indices = curr_ment_scores.topk(top_k)
	
			# Get top-k indices from approx-matrix
			approx_top_k_scores, approx_top_k_indices = approx_curr_ment_scores.topk(top_k_retvr)
			
			# Re-rank top-k indices from approx-matrix using exact scores from ment_to_ent matrix
			# Scores from given ment_to_ent matrix filled only for entities retrieved by approximate retriever
			temp = torch.zeros(curr_ment_scores.shape) - 99999999999999
			temp[approx_top_k_indices] = curr_ment_scores[approx_top_k_indices]
		
			top_k_w_approx_retrvr_scores, top_k_w_approx_retrvr_indices = temp.topk(top_k)
			
			topk_preds += [(top_k_indices.unsqueeze(0), top_k_scores.unsqueeze(0))]
			approx_topk_preds += [(approx_top_k_indices.unsqueeze(0), approx_top_k_scores.unsqueeze(0))]
			topk_w_approx_retrvr_preds += [(top_k_w_approx_retrvr_indices.unsqueeze(0), top_k_w_approx_retrvr_scores.unsqueeze(0))]
	
		
		topk_preds = _get_indices_scores(topk_preds)
		approx_topk_preds = _get_indices_scores(approx_topk_preds)
		topk_w_approx_retrvr_preds = _get_indices_scores(topk_w_approx_retrvr_preds)
		
		def score_topk_preds_wrapper(arg_ment_idxs):
			# exact_vs_approx_retvr = compute_overlap(
			# 	indices_list1=topk_preds["indices"][arg_ment_idxs],
			# 	indices_list2=approx_topk_preds["indices"][arg_ment_idxs],
			# )
			# new_exact_vs_approx_retvr = {}
			# for _metric in exact_vs_approx_retvr:
			# 	new_exact_vs_approx_retvr[f"{_metric}_mean"] = float(exact_vs_approx_retvr[_metric][0][5:])
			# 	new_exact_vs_approx_retvr[f"{_metric}_std"] = float(exact_vs_approx_retvr[_metric][1][4:])
			# 	new_exact_vs_approx_retvr[f"{_metric}_p50"] = float(exact_vs_approx_retvr[_metric][2][4:])
			
			exact_vs_reranked_approx_retvr = compute_overlap(
				indices_list1=topk_preds["indices"][arg_ment_idxs],
				indices_list2=topk_w_approx_retrvr_preds["indices"][arg_ment_idxs],
			)
			new_exact_vs_reranked_approx_retvr = {}
			for _metric in exact_vs_reranked_approx_retvr:
				new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
			
			# Compare overlap between anchor entities and exact top-k entities
			exact_vs_anchor_ents = compute_recall(
				gt_indices_list=topk_preds["indices"][arg_ment_idxs],
				pred_indices_list=np.array([anchor_ent_idxs for _ in arg_ment_idxs]),
			)
			
			new_exact_vs_anchor_ents = {}
			for _metric in exact_vs_anchor_ents:
				new_exact_vs_anchor_ents[f"{_metric}_mean"] = float(exact_vs_anchor_ents[_metric][0][5:])
				new_exact_vs_anchor_ents[f"{_metric}_std"] = float(exact_vs_anchor_ents[_metric][1][4:])
				new_exact_vs_anchor_ents[f"{_metric}_p50"] = float(exact_vs_anchor_ents[_metric][2][4:])
		
			res = {
				# f"exact_vs_approx_retrvr": new_exact_vs_approx_retvr,
				f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr,
				f"exact_vs_anchor_ents" : new_exact_vs_anchor_ents,
			}
			
			new_res = {}
			for res_type in res:
				for metric in res[res_type]:
					new_res[f"{res_type}~{metric}"] = res[res_type][metric]
			
			# new_res["approx_error"] =  (torch.norm( (approx_ment_to_ent_scores - all_ment_to_ent_scores)[arg_ment_idxs, :] )).data.numpy()
			# new_res["approx_error_relative"] =  new_res["approx_error"]/(torch.norm(all_ment_to_ent_scores[arg_ment_idxs, :]).data.numpy())
		
			new_res["approx_error_all"], new_res["approx_error_all_relative"], \
			new_res["approx_error_all_percent"], new_res["approx_error_all_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=np.array([np.arange(n_ents) for _ in arg_ment_idxs])
			)
			
			_, gt_top_k_ent_idxs = torch.topk(all_ment_to_ent_scores[arg_ment_idxs], k=top_k)
			new_res["approx_error_head"], new_res["approx_error_head_relative"], \
			new_res["approx_error_head_percent"], new_res["approx_error_head_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=gt_top_k_ent_idxs
			)
			
			_, gt_tail_ent_idxs = torch.topk(-1*all_ment_to_ent_scores[arg_ment_idxs], k=n_ents - top_k)
			new_res["approx_error_tail"], new_res["approx_error_tail_relative"], \
			new_res["approx_error_tail_percent"], new_res["approx_error_tail_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=gt_tail_ent_idxs
			)
			
			new_res["approx_error_anc_ents"], new_res["approx_error_anc_ents_relative"], \
			new_res["approx_error_anc_ents_percent"], new_res["approx_error_anc_ents_abs_percent"] = get_error(
				exact_mat=all_ment_to_ent_scores[arg_ment_idxs],
				approx_mat=approx_ment_to_ent_scores[arg_ment_idxs],
				cols_for_each_row=np.array([anchor_ent_idxs for _ in arg_ment_idxs])
			)
			

			return new_res
		
		final_res = {
			"anchor":score_topk_preds_wrapper(arg_ment_idxs=anchor_ment_idxs),
			"non_anchor":score_topk_preds_wrapper(arg_ment_idxs=non_anchor_ment_idxs),
			# "all":score_topk_preds_wrapper(arg_ment_idxs=list(range(n_ments)))
		}
		return final_res
	except Exception as e:
		embed()
		raise e



def run_approx_eval(approx_method, all_ment_to_ent_scores, precomp_approx_ment_to_ent_scores,
					n_ment_anchors, n_ent_anchors, top_k, top_k_retvr, n_seeds, icur_args):
	"""
	Run approximation eval for different seeds
	:param approx_method
	:param all_ment_to_ent_scores:
	:param precomp_approx_ment_to_ent_scores
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param top_k_retvr
	:param n_seeds:
	:return:
	"""
	
	avg_res = defaultdict(lambda : defaultdict(list))
	for seed in range(n_seeds):
		if approx_method == "i_cur":
			res = run_approx_eval_i_cur_w_seed(
				approx_method=approx_method,
				all_ment_to_ent_scores=all_ment_to_ent_scores,
				n_ment_anchors=n_ment_anchors,
				n_ent_anchors=n_ent_anchors,
				top_k=top_k,
				top_k_retvr=top_k_retvr,
				seed=seed,
				icur_args=icur_args
			)
		else:
			res = run_approx_eval_w_seed(
				approx_method=approx_method,
				all_ment_to_ent_scores=all_ment_to_ent_scores,
				precomp_approx_ment_to_ent_scores=precomp_approx_ment_to_ent_scores,
				n_ment_anchors=n_ment_anchors,
				n_ent_anchors=n_ent_anchors,
				top_k=top_k,
				top_k_retvr=top_k_retvr,
				seed=seed
			)
		
		# Accumulate results for each seed
		for ment_type, res_dict in res.items():
			for metric, val in res_dict.items():
				avg_res[ment_type][metric] += [float(val)]
	
	# Average results for all different seeds
	new_avg_res = defaultdict(dict)
	for ment_type in avg_res:
		for metric in avg_res[ment_type]:
			new_avg_res[ment_type][metric] = np.mean(avg_res[ment_type][metric])
	
	return new_avg_res
	

def run(base_res_dir, data_info, n_seeds, batch_size, plot_only, misc, disable_wandb, arg_dict, biencoder=None):
	
	try:
		
		if biencoder: biencoder.eval()
		data_name, data_fname = data_info
	
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			# test_data = dump_dict["test_data"]
			# entity_id_list = dump_dict["entity_id_list"]
			crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores[:2050, :]
			mention_tokens_list = mention_tokens_list[:2050]
		
		complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		mention_tokens_list = torch.LongTensor(mention_tokens_list)
		total_n_ment, total_n_ent = crossenc_ment_to_ent_scores.shape
		
		# eval_methods  = ["bienc", "cur"]
		# # eval_methods  = ["bienc", "cur", "fixed_anc_ent", "fixed_anc_ent_cur_100", "fixed_anc_ent_cur_1000"]
		# # eval_methods  = ["cur"]
		#
		# # n_ment_anchors_vals = [50,64,75,100,200, 500, 1000]
		# n_ment_anchors_vals = [50, 100, 200, 500, 1000, 2000, 3000]
		# n_ment_anchors_vals = [v for v in n_ment_anchors_vals if v <= total_n_ment]
		#
		# # n_ent_anchors_vals = [5, 10, 20, 50, 64, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
		# n_ent_anchors_vals = [50, 100, 200, 500, 1000]
		# n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v < total_n_ent] + [total_n_ent]
		#
		# # top_k_vals = [10,20,50,64,100,200]
		# # top_k_vals = [1, 10, 64]
		# top_k_vals = [1, 10, 20, 50, 100]
		# top_k_retr_vals = [1, 10, 20, 50, 100, 200, 500, 1000]
		# top_k_retr_vals_bienc = top_k_retr_vals + [x+y for x in top_k_retr_vals for y in n_ent_anchors_vals]
		# top_k_retr_vals_bienc = top_k_retr_vals
		
		
		# # For plots
		# eval_methods  = ["cur", "cur_oracle"]
		#
		# n_ment_anchors_vals = [50, 100, 200, 500, 1000, 2000, 5000]
		# # n_ment_anchors_vals = [50, 100, 200]
		# n_ment_anchors_vals = [v for v in n_ment_anchors_vals if v <= total_n_ment]
		#
		# n_ent_anchors_vals = [50, 100, 200, 500, 1000, 2000]
		# # n_ent_anchors_vals = [50, 100, 200]
		# n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v < total_n_ent] + [total_n_ent]
		#
		# top_k_vals = [1, 10, 50, 100]
		# top_k_retr_vals = [100, 500, 1000]
		# top_k_vals = [10]
		# top_k_retr_vals = [500]
		# top_k_retr_vals_bienc = top_k_retr_vals
		
		
		# For debug
		#
		# eval_methods  = ["fixed_anc_ent", "cur", "fixed_anc_ent_cur_100", "fixed_anc_ent_cur_1000"]
		# eval_methods  = ["cur", "fixed_anc_ent", "fixed_anc_ent_cur_100", "fixed_anc_ent_cur_1000"]
		# eval_methods  = ["i_cur"]
		
		# i_cur_params
		eval_methods  = ["i_cur"]
		# n_ment_anchors_vals = [v for v in [50, 500, 2000] if v < total_n_ment]
		n_ment_anchors_vals = [v for v in [2000] if v < total_n_ment]
		# n_ent_anchors_vals = [200, 800]
		n_ent_anchors_vals = [200]
		top_k_vals = [1, 100]
		top_k_retr_vals = [10, 50, 100, 200, 500, 1000]
		# top_k_retr_vals = [200, 500]
		top_k_retr_vals_bienc = top_k_retr_vals
		
		# # eval_methods  = ["svd"]
		# # eval_methods  = ["svd_oracle"]
		#
		# # eval_methods  = ["svd_linreg", "cur", "cur_linreg", "svd_linreg_well_cond"]
		# # eval_methods  = ["svd_sparse", "cur_w_svd_sparse"]
		# eval_methods  = ["cur_w_svd_sparse"]
		# # eval_methods  = ["cur"]
		# n_ment_anchors_vals = [v for v in [50, 100, 200, 500] if v < total_n_ment]
		# n_ent_anchors_vals = [50, 100, 200, 500]
		# top_k_vals = [100]
		# # top_k_retr_vals = [10, 50, 100, 200, 500, 1000]
		# top_k_retr_vals = [500]
		# top_k_retr_vals_bienc = top_k_retr_vals
		
		
		icur_args = {key: arg_dict[key] for key in ["shortlist_method", "sampling_method", "i_cur_n_steps"]}
		if "i_cur" in eval_methods:
			misc += "_".join([f"{k}={v}" for k,v in sorted(icur_args.items())])
		
		res_dir = f"{base_res_dir}/nm={total_n_ment}_ne={total_n_ent}_s={n_seeds}{misc}"
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		icur_args["res_dir"] = res_dir
		
		other_args = {'arg_dict': arg_dict, 'top_k_vals': top_k_vals, 'top_k_retr_vals': top_k_retr_vals,
					  'n_ent_anchors_vals': n_ent_anchors_vals, 'n_ment_anchors_vals': n_ment_anchors_vals}
			
		if not plot_only:
			wandb.init(
				project="Retrieval_WRT_Exact_CrossEnc",
				dir=res_dir,
				config=other_args,
				mode="disabled" if disable_wandb else "online"
			)
			
			eval_res = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
			for curr_method in eval_methods:
	
				LOGGER.info(f"Running inference for method = {curr_method}")
				precomp_approx_ment_to_ent_scores = {}
				if curr_method  == "bienc":
					label_embeds = compute_label_embeddings(
						biencoder=biencoder,
						labels_tokens_list=complete_entity_tokens_list,
						batch_size=batch_size
					)
	
					mention_embeds = compute_input_embeddings(
						biencoder=biencoder,
						input_tokens_list=mention_tokens_list,
						batch_size=batch_size
					)
					bienc_ment_to_ent_scores = mention_embeds @ label_embeds.T
					precomp_approx_ment_to_ent_scores = {x:bienc_ment_to_ent_scores for x in n_ent_anchors_vals}
				elif curr_method == "cur":
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method == "i_cur":
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method == "cur_oracle":
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method in ["svd_sparse", "svd_linreg", "svd_linreg_well_cond", "svd_oracle", "cur_linreg",
									 "cur_w_svd_sparse"]:
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method == "fixed_anc_ent":
					
					ment_to_ent_scores_wrt_anc_ents = {}
					ent2ent_dir = os.path.dirname(data_fname["crossenc_ment_to_ent_scores"])
					for n_anc_ent in n_ent_anchors_vals:
						e2e_fname = f"{ent2ent_dir}/ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_topk_{n_anc_ent}_embed_bienc_m2e_bienc_cluster.pkl"
						
						if not os.path.isfile(e2e_fname):
							LOGGER.info(f"File for n_anc_ent={n_anc_ent} not found")  #"i.e. {e2e_fname} does not exist")
							continue
						
						with open(e2e_fname, "rb") as fin:
							dump_dict = pickle.load(fin)
							ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : Number of entities x Number of anchors
							anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of anchors
							
						mention_embeds = crossenc_ment_to_ent_scores[:, anchor_ents]
						
						ment_to_ent_scores_wrt_anc_ents[n_anc_ent] = mention_embeds @ ent_embeds.T
						
						
					precomp_approx_ment_to_ent_scores = ment_to_ent_scores_wrt_anc_ents
					
				elif curr_method.startswith("fixed_anc_ent_cur_"):
					
					n_fixed_anchor_ents = int(curr_method[len("fixed_anc_ent_cur_"):])
					
					ent2ent_dir = os.path.dirname(data_fname["crossenc_ment_to_ent_scores"])
					e2e_fname = f"{ent2ent_dir}/ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_topk_{n_fixed_anchor_ents}_embed_bienc_m2e_bienc_cluster.pkl"
					
					if not os.path.isfile(e2e_fname):
						LOGGER.info(f"File for num_fixed_ent={n_fixed_anchor_ents} not found")  #"i.e. {e2e_fname} does not exist")
						continue
					
					with open(e2e_fname, "rb") as fin:
						dump_dict = pickle.load(fin)
						ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : (Number of entities, Number of anchors)
						fixed_anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of fixed anchor entities
						n_ents = ent_embeds.shape[0]
						
						R = ent_embeds.T # shape : (n_fixed_anchor_ents, n_ents)
						
					rng = np.random.default_rng(seed=0)
					ment_to_ent_scores_wrt_anc_ents = {}
					for n_anc_ent in n_ent_anchors_vals:
						
						anchor_ent_idxs = sorted(rng.choice(n_ents, size=n_anc_ent, replace=False))
						
						intersect_mat = R[:, anchor_ent_idxs] # shape: (n_fixed_anchor_ents, n_anc_ent)
						U = torch.tensor(np.linalg.pinv(intersect_mat))  # shape: (n_anc_ent, n_fixed_anchor_ents)
						UR = U @ R # shape: (n_anc_ent, n_ents)
						
						# Score of mentions w/ anchor entities,
						mention_embeds = crossenc_ment_to_ent_scores[:, anchor_ent_idxs] # shape: (n_ments, n_anc_ent)
						
						# (n_ments, n_ents) = (n_ments, n_anc_ent) x (n_anc_ent, n_ents)
						ment_to_ent_scores_wrt_anc_ents[n_anc_ent] = mention_embeds @ UR
						
						
					precomp_approx_ment_to_ent_scores = ment_to_ent_scores_wrt_anc_ents
				
				else:
					raise NotImplementedError(f"Method = {curr_method} not supported")
				
				if curr_method == "bienc":
					val_tuple_list = list(itertools.product(top_k_vals, top_k_retr_vals_bienc, n_ment_anchors_vals, n_ent_anchors_vals))
				else:
					val_tuple_list = list(itertools.product(top_k_vals, top_k_retr_vals, n_ment_anchors_vals, n_ent_anchors_vals))
					
				for ctr, (top_k, top_k_retvr, n_ment_anchors, n_ent_anchors) in tqdm(enumerate(val_tuple_list), total=len(val_tuple_list)):
					wandb.log({f"ctr_{curr_method}":ctr/len(val_tuple_list)})
					if top_k_retvr < top_k: continue
					if top_k_retvr > total_n_ent: continue
					if n_ent_anchors not in precomp_approx_ment_to_ent_scores: continue
					
					if curr_method == "bienc" and n_ent_anchors != n_ent_anchors_vals[0]:
						prev_ans = eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors_vals[0]}"]
						eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = prev_ans
						continue
						
					if curr_method.startswith("bienc") and n_ment_anchors != n_ment_anchors_vals[0]:
						prev_ans = eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors_vals[0]}~anc_n_e={n_ent_anchors}"]
						eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = prev_ans
						continue
						
					curr_ans = run_approx_eval(
						approx_method=curr_method,
						all_ment_to_ent_scores=crossenc_ment_to_ent_scores,
						precomp_approx_ment_to_ent_scores=precomp_approx_ment_to_ent_scores[n_ent_anchors],
						n_ment_anchors=n_ment_anchors,
						n_ent_anchors=n_ent_anchors,
						top_k=top_k,
						top_k_retvr=top_k_retvr,
						n_seeds=n_seeds,
						icur_args=icur_args
					)
	
					eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = curr_ans
	
		
			res_fname = f"{res_dir}/retrieval_wrt_exact_crossenc.json"
			eval_res["other_args"] = other_args
			with open(res_fname, "w") as fout:
				json.dump(obj=eval_res, fp=fout, indent=4)


		# plot(
		# 	res_dir=res_dir,
		# 	method_vals=eval_methods,
		# 	# n_ment_anchors_vals=n_ment_anchors_vals,
		# 	# n_ent_anchors_vals=n_ent_anchors_vals,
		# 	# top_k_retvr_vals=top_k_retr_vals,
		# 	# top_k_vals=top_k_vals,
		# 	# method_vals=eval_methods
		# )
		
	except Exception as e:
		embed()
		raise e
	



def plot(res_dir, method_vals):
	
	try:
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]
		LOGGER.info("Now plotting results")
		res_fname = f"{res_dir}/retrieval_wrt_exact_crossenc.json"
		with open(res_fname, "r") as fin:
			eval_res = json.load(fp=fin)
	
		n_ment_anchors_vals = eval_res["other_args"]["n_ment_anchors_vals"]
		n_ent_anchors_vals = eval_res["other_args"]["n_ent_anchors_vals"]
		top_k_vals = eval_res["other_args"]["top_k_vals"]
		top_k_retvr_vals = eval_res["other_args"]["top_k_retr_vals"]
		n_ent = NUM_ENTS[eval_res["other_args"]["arg_dict"]["data_name"]]
		
		############################# NOW VISUALIZE RESULTS AS A FUNCTION OF N_ANCHORS ####################################
		# metrics = [f"exact_vs_reranked_approx_retvr~{m1}_{m2}" for m1 in ["common", "diff", "total", "common_frac", "diff_frac"] for m2 in ["mean", "std", "p50"]]
		metrics = [f"exact_vs_reranked_approx_retvr~common_frac_mean"]
		metrics += [f"approx_error{ent_type}_relative" for ent_type in ["_all", "_head", "_tail", "_anc_ents"]]
		metrics += [f"approx_error{ent_type}_percent" for ent_type in ["_all", "_head", "_tail", "_anc_ents"]]
		# metrics += [f"approx_error{ent_type}" for ent_type in ["_all", "_head", "_tail", "_anc_ents"]]
		
	
		# mtype_vals = ["non_anchor", "all", "anchor"]
		mtype_vals = ["non_anchor"]
		for mtype, curr_method, top_k, top_k_retvr, metric in itertools.product(mtype_vals, method_vals, top_k_vals, top_k_retvr_vals, metrics):
			if top_k > top_k_retvr: continue
			val_matrix = []
			# Build matrix for given topk value with varying number of anchor mentions and anchor entities
			try:
				for n_ment_anchors in n_ment_anchors_vals:
					# curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
					curr_config_res = [100*eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
									   if f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}" in eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"]
									   else 0.
									   for n_ent_anchors in n_ent_anchors_vals]
					val_matrix += [curr_config_res]
			except KeyError as e:
				LOGGER.info(f"Key-error = {e} for mtype = {mtype}, curr_method={curr_method}, top_k={top_k}, top_k_retvr={top_k_retvr}, metric={metric}")
				# embed()
				continue
	
			val_matrix = np.array(val_matrix, dtype=np.float64)
			curr_res_dir = f"{res_dir}/plots_{mtype}/k={top_k}/separate_plots/k_retr={top_k_retvr}_{curr_method}"
			Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
			
			plot_heat_map(
				val_matrix=val_matrix,
				row_vals=n_ment_anchors_vals,
				col_vals=n_ent_anchors_vals,
				metric=metric,
				top_k=top_k,
				curr_res_dir=curr_res_dir
			)
	
		# # Plot results for CUR relative to bienc method
		# for mtype, top_k, top_k_retvr, metric in itertools.product(mtype_vals, top_k_vals, top_k_retvr_vals, metrics):
		# 	if top_k > top_k_retvr: continue
		# 	val_matrix_1 = []
		# 	val_matrix_2 = []
		# 	# Build matrix for given topk value with varying number of anchor mentions and anchor entities
		# 	try:
		# 		for n_ment_anchors in n_ment_anchors_vals:
		# 			# curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
		# 			curr_config_res_cur = [
		# 				eval_res["cur"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
		# 				for n_ent_anchors in n_ent_anchors_vals
		# 			]
		# 			curr_config_res_bienc = [
		# 				eval_res["bienc"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
		# 				for n_ent_anchors in n_ent_anchors_vals
		# 			]
		# 			# curr_config_res_bienc_same_cost = curr_config_res_bienc
		# 			curr_config_res_bienc_same_cost = [
		# 				eval_res["bienc"][f"top_k={top_k}"][f"k_retvr={top_k_retvr+n_ent_anchors}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
		# 				if top_k_retvr+n_ent_anchors < n_ent
		# 				else eval_res["bienc"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
		# 				for n_ent_anchors in n_ent_anchors_vals
		# 			]
		#
		# 			val_matrix_1 += [[x-y for x,y in zip(curr_config_res_cur, curr_config_res_bienc)]]
		# 			val_matrix_2 += [[x-y for x,y in zip(curr_config_res_cur, curr_config_res_bienc_same_cost)]]
		# 	except KeyError as e:
		# 		LOGGER.info(f"Key-error = {e} for mtype = {mtype}, top_k={top_k}, top_k_retvr={top_k_retvr}, metric={metric}")
		# 		# embed()
		# 		continue
		#
		# 	val_matrix_1 = np.array(val_matrix_1, dtype=np.float64)
		# 	val_matrix_2 = np.array(val_matrix_2, dtype=np.float64)
		# 	curr_res_dir = f"{res_dir}/plots_{mtype}/k={top_k}/separate_plots/k_retr={top_k_retvr}_cur"
		# 	Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		#
		# 	# LOGGER.info(f"Saving in res_dir {curr_res_dir}")
		# 	plot_heat_map(
		# 		val_matrix=val_matrix_1,
		# 		row_vals=n_ment_anchors_vals,
		# 		col_vals=n_ent_anchors_vals,
		# 		metric=metric,
		# 		top_k=top_k,
		# 		curr_res_dir=curr_res_dir,
		# 		fname=f"{metric}_same_num_retrieved"
		# 	)
		#
		# 	plot_heat_map(
		# 		val_matrix=val_matrix_2,
		# 		row_vals=n_ment_anchors_vals,
		# 		col_vals=n_ent_anchors_vals,
		# 		metric=metric,
		# 		top_k=top_k,
		# 		curr_res_dir=curr_res_dir,
		# 		fname=f"{metric}_same_num_crossenc_calls"
		# 	)
	
	
		################################################################################################################
		
		
		# LOGGER.info("Now plotting recall-vs-cost plots for comparing bienc and cur method")
		# ################################################################################################################
		# mtype_vals = ["non_anchor"]
		# metrics = [f"exact_vs_reranked_approx_retvr~{m1}_{m2}" for m1 in ["common_frac"] for m2 in ["mean"]]
		#
		# for mtype, top_k, metric in itertools.product(mtype_vals, top_k_vals, metrics):
		# 	try:
		# 		plt.clf()
		# 		y_vals = defaultdict(list)
		# 		x_vals = defaultdict(list)
		#
		#
		# 		for n_ment_anchors, n_ent_anchors, top_k_retvr in itertools.product(n_ment_anchors_vals, n_ent_anchors_vals, top_k_retvr_vals):
		# 			if top_k > top_k_retvr: continue
		# 			# curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
		# 			y_vals["cur"] += [eval_res["cur"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]]
		# 			y_vals["bienc"] += [eval_res["bienc"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]]
		#
		# 			x_vals["cur"] += [top_k_retvr + n_ent_anchors]
		# 			x_vals["bienc"] += [top_k_retvr]
		#
		#
		# 		plt.scatter(x_vals["bienc"], y_vals["cur"], c=colors[2][1], label="cur-wo-anc-cost", alpha=0.5,edgecolors=colors[2][0])
		# 		plt.scatter(x_vals["cur"], y_vals["cur"], c=colors[1][1], label="cur", alpha=0.5,edgecolors=colors[1][0])
		# 		plt.scatter(x_vals["bienc"], y_vals["bienc"], c=colors[0][1], label="bienc", alpha=0.5, edgecolors=colors[0][0])
		#
		# 		plt.xlim(1, 1100)
		# 		plt.legend()
		# 		plt.grid()
		# 		plt.xlabel("Cost")
		# 		plt.ylabel("Recall")
		# 		curr_plt_file = f"{res_dir}/plots_{mtype}/k={top_k}/recall_vs_cost/{metric}.pdf"
		# 		Path(os.path.dirname(curr_plt_file)).mkdir(exist_ok=True, parents=True)
		# 		plt.savefig(curr_plt_file)
		#
		# 		plt.xscale("log")
		# 		curr_plt_file = f"{res_dir}/plots_{mtype}/k={top_k}/recall_vs_cost/{metric}_xlog.pdf"
		# 		plt.savefig(curr_plt_file)
		#
		# 		plt.close()
		#
		# 	except KeyError as e:
		# 		LOGGER.info(f"Key-error = {e} for mtype = {mtype}, top_k={top_k}, metric={metric}")
		# 		continue
		#
	
	except Exception as e:
		embed()
		raise e
		


def plot_score_distribution(out_dir, score_matrices, anchor_ent_idxs, bins=200, num_rows=100):
	
	try:
		mat_files = {
			"approx" : {
				"color":"lightgreen",
				"label":"Approx",
			},
			"exact" 		: {
				"color":"red",
				"label":"Exact",
			}
		}
		
		plt.clf()
		fig, ax = plt.subplots(figsize=(10,8))
		
		for score_matrix_type, score_matrix in score_matrices.items():
			score_matrix = score_matrix[:num_rows].reshape(-1)
			# score_matrix = score_matrix - np.mean(score_matrix) # Center around zero by subtracting mean
			
			sns.distplot(score_matrix,
						 hist = True, kde = True,
						 kde_kws={'shade': True, 'linewidth': 2},
						 bins=bins,
						 color=mat_files[score_matrix_type]["color"],
						 ax=ax, label=mat_files[score_matrix_type]["label"])
			
		ax.set_xlabel("Query-Item Score", fontsize=50)
		ax.set_ylabel("Score Density", fontsize=50)
		# ax.set_xlim(-15,15)
		ax.tick_params(axis='both', which='major', labelsize=40)
		plt.legend(prop={'size': 30})
		fig.tight_layout()
		
		out_filename = f"{out_dir}/score_dist_n={num_rows}_joint_bins={bins}.pdf"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename, bbox_inches='tight')
		
		plt.yscale('log')
		out_filename = f"{out_dir}/score_dist_n={num_rows}_joint_log_scale_bins={bins}.pdf"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename, bbox_inches='tight')
	
		
		
		# # for ylim in [0.001, 0.01, 0.1]:
		# for ylim in [0.001]:
		# 	plt.yscale('linear')
		# 	plt.gca().set_ylim(bottom=ylim)
		# 	out_filename = f"{out_dir}/score_dist_n={num_rows}_joint_linear_scale_ylim_{ylim}_bins={bins}.pdf"
		# 	Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		# 	plt.savefig(out_filename, bbox_inches='tight')
		#
		# 	plt.yscale('log')
		# 	plt.gca().set_ylim(bottom=ylim)
		# 	out_filename = f"{out_dir}/score_dist_n={num_rows}_joint_log_scale_ylim_{ylim}_bins={bins}.pdf"
		# 	Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		# 	plt.savefig(out_filename, bbox_inches='tight')
		
		plt.close()
		
		# Scatter plot of exact-vs-approx scores
		## Plot for each mentions in a separate plot
		approx_score_mat = score_matrices["approx"]
		exact_score_mat = score_matrices["exact"]
		
		for i, (X,Y) in enumerate(zip(exact_score_mat[:10], approx_score_mat[:10])):
			plt.clf()
			### Plot all entities
			plt.scatter(X,Y, marker='x', alpha=0.5, cmap=plt.get_cmap("Spectral"), label="all entities")
			
			### Plot anchor entities
			plt.scatter(X[anchor_ent_idxs[i]],Y[anchor_ent_idxs[i]], marker='o', alpha=0.5, cmap=plt.get_cmap("Reds"), label="anchor entities")
			
			min_X, min_Y = min(X), min(Y)
			max_X, max_Y = max(X), max(Y)
			
			guide_X = [ max(min_X, min_Y), min(max_X, max_Y)]
			plt.plot(guide_X, guide_X, c="k")
			
			plt.legend()
			plt.xlabel("Exact Scores")
			plt.ylabel("Approx Scores")
		
			out_filename = f"{out_dir}/score_dist_n={num_rows}_scatter/ment_idx={i}.png"
			Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
			plt.savefig(out_filename)
		
		## Plot for all mentions in a single plot
		plt.clf()
		plt.xlabel("Exact Scores")
		plt.ylabel("Approx Scores")
	
		### Plot all entities
		for X,Y in zip(exact_score_mat[:num_rows], approx_score_mat[:num_rows]):
			plt.scatter(X,Y, marker='x', alpha=0.5, cmap=plt.get_cmap("Spectral"))
		guide_X = [max(np.min(exact_score_mat[:num_rows]), np.min(approx_score_mat[:num_rows])),
				   min(np.max(exact_score_mat[:num_rows]), np.max(approx_score_mat[:num_rows]))]
		plt.plot(guide_X, guide_X, c="k")
		
		out_filename = f"{out_dir}/score_dist_n={num_rows}_scatter_combined_wo_anchors.png"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename)
		
		
		### Plot anchor entities
		for i, (X,Y) in enumerate(zip(exact_score_mat[:num_rows], approx_score_mat[:num_rows])):
			plt.scatter(X[anchor_ent_idxs[i]],Y[anchor_ent_idxs[i]], marker='o', alpha=0.5, cmap=plt.get_cmap("Spectral"))
		
		plt.plot(guide_X, guide_X, c="k")
		
		out_filename = f"{out_dir}/score_dist_n={num_rows}_scatter_combined.png"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename)
		plt.close()
		
		
	except Exception as e:
		embed()
		raise e
	
	
def main():
	
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run eval for various retrieval methods wrt exact crossencoder scores. This evaluation does not use ground-truth entity information into account')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	
	parser.add_argument("--bi_model_file", type=str, default="", help="File for biencoder ckpt")
	parser.add_argument("--res_dir", type=str, required=True, help="Res dir with score matrices, and to save results")
	parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to run")
	parser.add_argument("--plot_only", type=int, default=0, choices=[0,1], help="1 to only plot results, 0 to run exp and then plot results")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions in precomputed mention-entity score matrix")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size to use with biencoder")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	# Special args for incremental CUR method
	parser.add_argument("--shortlist_method", type=str, default="none", choices=["approx", "exact", "none"], help="Type of scores to use for shortlisting entities in each step of incremental CUR approach")
	parser.add_argument("--sampling_method", type=str, default="random_cumul",
						choices=[
							"random_cumul", "random_diff",
							"variance_cumul", "variance_topk_cumul"
							"approx_cumul", "approx_softmax_cumul", "approx_topk_cumul",
							"exact_cumul", "exact_softmax_cumul", "exact_topk_cumul", "exact_after_topk_cumul",
							"exact_diff", "exact_softmax_diff", "approx_diff", "approx_softmax_diff",
						],
						help="Sampling method to use for finding anchor entities, this also tells us if we should use all anchors found up to round t or shoull we use disjoint set of anchors in each step")
	parser.add_argument("--i_cur_n_steps", type=int, default=2, help="Number of steps for incremental CUR")
	

	args = parser.parse_args()
	data_name = args.data_name
	
	bi_model_file = args.bi_model_file
	res_dir = args.res_dir
	n_seeds = args.n_seeds
	n_ment = args.n_ment
	batch_size = args.batch_size
	plot_only = bool(args.plot_only)
	disable_wandb = bool(args.disable_wandb)
	misc = "_" + args.misc if args.misc != "" else ""
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	
	try:
		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file) if bi_model_file != "" else None
	except:
		LOGGER.info("Loading biencoder which was trained while sharing parameters with a cross-encoder.")
		biencoder = CrossEncoderWrapper.load_from_checkpoint(bi_model_file) if bi_model_file != "" else None
	
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		base_res_dir=f"{res_dir}/{data_name}/Retrieval_wrt_Exact_CrossEnc",
		data_info=(data_name, DATASETS[data_name]),
		n_seeds=n_seeds,
		batch_size=batch_size,
		plot_only=plot_only,
		biencoder=biencoder,
		disable_wandb=disable_wandb,
		misc=misc,
		arg_dict=args.__dict__
	)


if __name__ == "__main__":
	main()

