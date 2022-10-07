import os
import sys
import copy
import json
import torch
import pickle
import logging
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from IPython import embed
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from numpy.linalg import matrix_rank

from eval.eval_utils import score_topk_preds, compute_label_embeddings, compute_overlap
from eval.visualize_matrix_zeshel import visualize_hist
from models.biencoder import BiEncoderWrapper
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)




class CURApprox(object):

	def __init__(self, rows, cols, row_idxs, col_idxs, approx_preference, A=None):

		# print("\n\n\n\nThis is biased towards giving better approximations for new rows or cols -- think about it\n\n\n")
		super(CURApprox, self).__init__()
		# M :(n x m) = C U R : (n x kc) X (kc x kr) X (kr x m)

		self.n = cols.shape[0]
		self.m = rows.shape[1]

		self.row_idxs = row_idxs
		self.col_idxs = col_idxs

		self.C = cols # n x kc
		self.R = rows # kr x m
		
		self.approx_preference = approx_preference
		assert self.approx_preference in ["rows", "cols"], f"self.approx_preference = {self.approx_preference} but should be  one of [rows, cols]"
		assert self._is_sorted(self.row_idxs), "row_idxs should be sorted"
		assert self._is_sorted(self.col_idxs), "col_idxs should be sorted"

		assert len(row_idxs) == self.R.shape[0]
		assert len(col_idxs) == self.C.shape[1]

		intersect_mat = self.C[row_idxs, :] # kr x kc

		# a = torch.eq(self.C[row_idxs, :], self.R[:, col_idxs])
		# print(a)
		# assert torch.eq(self.C[row_idxs, :], self.R[:, col_idxs]), "Invalid rows and cols as their intersection does not match"
		
		# if len(row_idxs) == len(col_idxs):
		# 	try:
		# 		isvd_u, isvd_s, isvd_vt = np.linalg.svd(intersect_mat, full_matrices=True)
		# 		s_inv = torch.tensor(np.diag(1/isvd_s))
		# 		isvd_v = torch.tensor(isvd_vt).T
		# 		isvd_ut = torch.tensor(isvd_u).T
		#
		# 		self.U = isvd_v @ s_inv @ isvd_ut
		# 		cond_num  =
		# 		# self.U = torch.tensor(np.linalg.inv(intersect_mat)) # kc x kr
		# 		LOGGER.info(f"Square intersection cond num: {cond_num:.2f}")
		# 	except Exception as e:
		# 		embed()
		# 		raise e
		# else:
		# 	self.U = torch.tensor(np.linalg.pinv(intersect_mat)) # kc x kr
		# 	cond_num  = np.linalg.cond(intersect_mat)
		# 	LOGGER.info(f"Rectangular intersection cond num: {cond_num:.2f}")
		
		
		# SMS-Nystrom idea from Archan's paper - did not work
		# if len(row_idxs) == len(col_idxs):
		# 	isvd_u, isvd_s, isvd_vt = np.linalg.svd(intersect_mat, full_matrices=True)
		# 	min_singular_val = min(isvd_s)
		# 	alpha = 1.5
		# 	new_intersect_mat = intersect_mat - alpha*min_singular_val*np.eye(len(row_idxs),dtype=np.float32)
		#
		# 	LOGGER.info(f"Orig Square intersection cond num: {np.linalg.cond(intersect_mat):.2f}")
		# 	LOGGER.info(f"New Square intersection cond num: {np.linalg.cond(new_intersect_mat):.2f}")
		# 	self.U = torch.tensor(np.linalg.pinv(new_intersect_mat)) # kc x kr
		# else:
		# 	self.U = torch.tensor(np.linalg.pinv(intersect_mat)) # kc x kr
		
		
		if A is not None: # A better conditioned way of estimating U matrix but this requires computing entire matrix A ahead of time.
			self.U = torch.tensor(np.linalg.pinv(self.C)) @ torch.tensor(A) @ torch.tensor(np.linalg.pinv(self.R))
		else:
			self.U = torch.tensor(np.linalg.pinv(intersect_mat)) # kc x kr
			
		self.latent_rows, self.latent_cols = self._build_latent_row_cols(C=self.C, U=self.U, R=self.R, approx_preference=self.approx_preference)

	@staticmethod
	def _is_sorted(idx_list):
		return all(i < j for i,j in zip(idx_list[:-1], idx_list[1:]))

	@staticmethod
	def _build_latent_row_cols(C, U, R, approx_preference):

		if approx_preference == "cols":
			latent_rows = C @ U # n x kr
			latent_cols = R # kr x m
		elif approx_preference == "rows":
			latent_rows = C # n x kc
			latent_cols = U @ R # kc x m
		else:
			raise NotImplementedError(f"approx_preference = {approx_preference} not supported")
		
		return latent_rows, latent_cols

	def get_rows(self, row_idxs):

		# len(row_idxs) x m) =  (len(row_idxs) x kr) X ( kr, m))
		ans = self.latent_rows[row_idxs,:] @ self.latent_cols
		return ans

	def get_cols(self, col_idxs):
		# n x len(col_idxs) =  (n x kr) X (kr, len(col_idxs))
		ans = self.latent_rows @ self.latent_cols[:, col_idxs]
		return ans

	def get(self, row_idxs, col_idxs):

		# len(row_idxs) x len(col_idxs) =  (len(row_idxs) x kr) X ( kr, len(col_idxs))
		ans = self.latent_rows[row_idxs,:] @ self.latent_cols[:, col_idxs]
		return ans

	def get_complete_col(self, sparse_cols):
		"""
		Take values in cols corresponding to anchor row indices and return complete cols
		:param sparse_cols:
		:return:
		"""
		if self.approx_preference != "cols":
			raise NotImplementedError("This is not designed to give good approx of cols as U matrix is multiplied w/ R matrix. Build index w/ approx_preference = cols instead.")
		# (n x *) = (n x kr) X (kr x *)
		dense_cols = self.latent_rows @ sparse_cols
		return dense_cols

	def topk_in_col(self, sparse_cols, k):
		"""
		Return top-k indices in these col(s)
		:return:
		"""

		return torch.topk(self.get_complete_col(sparse_cols=sparse_cols), k, dim=1)


	def get_complete_row(self, sparse_rows):
		"""
		Take values in rows corresponding to anchor col indices and return complete rows
		:param sparse_cols:
		:return:
		"""
		if self.approx_preference != "rows":
			raise NotImplementedError("This is not designed to give good approx of rows as C and U matrix are multiplied together. Build index w/ approx_preference = rows instead.")
		# (* x m) = (* x kr) X (kr x m)
		dense_rows = sparse_rows @ self.latent_cols
		return dense_rows

	def topk_in_row(self, sparse_rows, k):
		"""
		Return top-k indices in these row(s)
		:return:
		"""
		return torch.topk(self.get_complete_row(sparse_rows=sparse_rows), k, dim=1)


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


def run_approx_eval_w_seed(all_ment_to_ent_scores, gt_labels, n_ment_anchors, n_ent_anchors, top_k, seed):
	"""
	Takes a mention x entity matrix as input, uses some rows and columns of this matrix to approximate the entire matrix
	and also evaluate the approximation
	:param all_ment_to_ent_scores:
	:param gt_labels:
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param seed:
	:return:
	"""

	
	try:
		# LOGGER.info("Starting approx eval")
		
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
	
		approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows")
		approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
	
		topk_preds = []
		approx_topk_preds = []
		topk_w_approx_retrvr_preds = []
		
		
		# Path(res_dir).mkdir(exist_ok=True, parents=True)
		# visualize_hist(val_matrix=approx_ment_to_ent_scores[anchor_ment_idxs, :],
		# 			   title=f"anchor_top_k_{top_k}_approx",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels[anchor_ment_idxs])
		#
		# visualize_hist(val_matrix=approx_ment_to_ent_scores[non_anchor_ment_idxs, :],
		# 			   title=f"non_anchor_top_k_{top_k}_approx",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels[non_anchor_ment_idxs])
		#
		# visualize_hist(val_matrix=approx_ment_to_ent_scores,
		# 			   title=f"all_top_k_{top_k}_approx",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels)
		#
		# visualize_hist(val_matrix=all_ment_to_ent_scores[anchor_ment_idxs, :],
		# 			   title=f"anchor_top_k_{top_k}_exact",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels[anchor_ment_idxs])
		#
		# visualize_hist(val_matrix=all_ment_to_ent_scores[non_anchor_ment_idxs, :],
		# 			   title=f"non_anchor_top_k_{top_k}_exact",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels[non_anchor_ment_idxs])
		#
		# visualize_hist(val_matrix=all_ment_to_ent_scores,
		# 			   title=f"all_top_k_{top_k}_exact",
		# 			   curr_res_dir=res_dir,
		# 			   topk=top_k,
		# 			   gt_labels=gt_labels)
		
		
		for ment_idx in range(n_ments):
	
			curr_ment_scores = all_ment_to_ent_scores[ment_idx]
			approx_curr_ment_scores = approx_ment_to_ent_scores[ment_idx]
	
			# Get top-k indices from exact matrix
			top_k_scores, top_k_indices = curr_ment_scores.topk(top_k)
	
			# Get top-k indices from approx-matrix
			approx_top_k_scores, approx_top_k_indices = approx_curr_ment_scores.topk(top_k)
			
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
			exact_vs_approx_retvr = compute_overlap(
				indices_list1=topk_preds["indices"][arg_ment_idxs],
				indices_list2=approx_topk_preds["indices"][arg_ment_idxs],
			)
			new_exact_vs_approx_retvr = {}
			for _metric in exact_vs_approx_retvr:
				new_exact_vs_approx_retvr[f"{_metric}_mean"] = float(exact_vs_approx_retvr[_metric][0][5:])
				new_exact_vs_approx_retvr[f"{_metric}_std"] = float(exact_vs_approx_retvr[_metric][1][4:])
				new_exact_vs_approx_retvr[f"{_metric}_p50"] = float(exact_vs_approx_retvr[_metric][2][4:])
				
			res = {
				"exact": score_topk_preds(
					gt_labels=gt_labels[arg_ment_idxs],
					topk_preds={
						"indices":topk_preds["indices"][arg_ment_idxs],
						"scores":topk_preds["scores"][arg_ment_idxs]
					}
				),
				"approx": score_topk_preds(
					gt_labels=gt_labels[arg_ment_idxs],
					topk_preds={
						"indices":approx_topk_preds["indices"][arg_ment_idxs],
						"scores":approx_topk_preds["scores"][arg_ment_idxs]
					}
				),
				"exact_w_approx_retrvr": score_topk_preds(
					gt_labels=gt_labels[arg_ment_idxs],
					topk_preds={
						"indices":topk_w_approx_retrvr_preds["indices"][arg_ment_idxs],
						"scores":topk_w_approx_retrvr_preds["scores"][arg_ment_idxs]
					}
				),
				f"exact_vs_approx_retrvr": new_exact_vs_approx_retvr,
			}
			new_res = {}
			for res_type in res:
				for metric in res[res_type]:
					new_res[f"{res_type}~{metric}"] = res[res_type][metric]
			
			new_res["approx_error"] =  (torch.norm( (approx_ment_to_ent_scores - all_ment_to_ent_scores)[arg_ment_idxs, :] )).data.numpy()
			return new_res
		
		final_res = {
			"anchor":score_topk_preds_wrapper(arg_ment_idxs=anchor_ment_idxs),
			"non_anchor":score_topk_preds_wrapper(arg_ment_idxs=non_anchor_ment_idxs),
			"all":score_topk_preds_wrapper(arg_ment_idxs=list(range(n_ments)))
		}
		return final_res
	except Exception as e:
		embed()
		raise e
	

def run_approx_eval(all_ment_to_ent_scores, gt_labels, n_ment_anchors, n_ent_anchors, top_k, n_seeds):
	"""
	Run matrix approximation eval for different seeds
	:param all_ment_to_ent_scores:
	:param gt_labels
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param n_seeds:
	:param res_dir
	:return:
	"""
	try:
		avg_res = defaultdict(lambda : defaultdict(list))
		for seed in range(n_seeds):
			res = run_approx_eval_w_seed(
				all_ment_to_ent_scores=all_ment_to_ent_scores,
				gt_labels=gt_labels,
				n_ment_anchors=n_ment_anchors,
				n_ent_anchors=n_ent_anchors,
				top_k=top_k,
				seed=seed,
				# res_dir=f"{res_dir}/seed_{seed}"
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
	except Exception as e:
		embed()
		raise e


def run_approx_eval_for_diff_n_anchors(ment_to_ent_scores, gt_labels, res_dir, n_seeds, arg_dict):
	"""
	Evaluate matrix approximation for different number of anchor mentions and entities
	
	:return:
	"""
	try:
		############################### COMPUTE RESULT FOR ALL METHODS #################################################
		
		LOGGER.info(f"Rank of matrix of dim = {ment_to_ent_scores.shape} is {matrix_rank(ment_to_ent_scores)}")
		
		total_n_ment, total_n_ent = ment_to_ent_scores.shape
		
		n_ment_anchors_vals = [5,10,20,50,64,75,90,100,200, 400, 500, 800, 900, 1000]
		n_ment_anchors_vals = [v for v in n_ment_anchors_vals if v <= total_n_ment]

		n_ent_anchors_vals = [5, 10, 20, 50, 64, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
		n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v < total_n_ent]
		n_ent_anchors_vals = n_ent_anchors_vals + [total_n_ent]

		# top_k_vals = [10,20,50,64,100,200]
		top_k_vals = [1, 64]
		
		# Values to use for debugging
		# n_ment_anchors_vals = [50]
		# n_ent_anchors_vals = [100]
		# top_k_vals = [100]

		eval_res = {
			"n_ment_anchors_vals": n_ment_anchors_vals,
			"n_ent_anchors_vals": n_ent_anchors_vals,
			"top_k_vals": top_k_vals,
			"total_n_ment": total_n_ment,
			"total_n_ent": total_n_ent,
			"arg_dict": arg_dict
		}

		for top_k in tqdm(top_k_vals):
			for n_ment_anchors, n_ent_anchors in itertools.product(n_ment_anchors_vals, n_ent_anchors_vals):
				curr_ans = run_approx_eval(
					all_ment_to_ent_scores=ment_to_ent_scores,
					gt_labels=gt_labels,
					n_ment_anchors=n_ment_anchors,
					n_ent_anchors=n_ent_anchors,
					top_k=top_k,
					n_seeds=n_seeds,
					# res_dir=f"{res_dir}/hist"
				)
				eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"] = curr_ans

		Path(res_dir).mkdir(exist_ok=True, parents=True)
		res_fname = f"{res_dir}/matrix_approx_res.json"
		with open(res_fname, "w") as fout:
			json.dump(obj=eval_res, fp=fout, indent=4)

		################################################################################################################
		
		
		########################################## NOW VISUALIZE RESULTS ###############################################
		LOGGER.info("Now plotting results")
		res_fname = f"{res_dir}/matrix_approx_res.json"
		with open(res_fname, "r") as fin:
			eval_res = json.load(fp=fin)
			
		res_types = ["exact", "approx", "exact_w_approx_retrvr"]
		eval_metrics = ["acc", "mrr", "recall", "recall_5", "recall_10", "norm_acc", "norm_mrr"]

		metrics = [f"{res_type}~{eval_metric}" for (res_type, eval_metric) in itertools.product(res_types, eval_metrics)]
		metrics += ["approx_error"]
		metrics += [f"exact_vs_approx_retrvr~{m1}_{m2}" for m1 in ["common", "diff", "total", "common_frac", "diff_frac"] for m2 in ["mean", "std", "p50"]]

		top_k_vals = [1, 64] #
		for mtype in ["anchor", "non_anchor", "all"]:
			for top_k, metric in itertools.product(top_k_vals, metrics):
				val_matrix = []
				# Build matrix for given topk value with varying number of anchor mentions and anchor entities
				for n_ment_anchors in n_ment_anchors_vals:
					curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
					val_matrix += [curr_config_res]

				val_matrix = np.array(val_matrix, dtype=np.float64)
				curr_res_dir = f"{res_dir}/plots_{mtype}/k={top_k}"
				Path(curr_res_dir).mkdir(exist_ok=True, parents=True)

				plot_heat_map(
					val_matrix=val_matrix,
					row_vals=n_ment_anchors_vals,
					col_vals=n_ent_anchors_vals,
					metric=metric,
					top_k=top_k,
					curr_res_dir=curr_res_dir
				)

		################################################################################################################
			
	except Exception as e:
		embed()
		raise e
		
		
def plot_heat_map(val_matrix, row_vals, col_vals, metric, top_k, curr_res_dir, title=None, fname=None):
	"""
	Plot a heat map using give matrix and add x-/y-ticks and title
	:param val_matrix: Matrix for plotting heat map
	:param row_vals: y-ticks
	:param col_vals: x-ticks
	:param metric:
	:param top_k:
	:param curr_res_dir:
	:return:
	"""
	try:
		plt.clf()
		try:
			if np.max(val_matrix) > 100:
				fig, ax = plt.subplots(figsize=(12,12))
			else:
				fig, ax = plt.subplots(figsize=(8,8))
		except:
			fig, ax = plt.subplots(figsize=(8,8))
			
		im = ax.imshow(val_matrix)
		
	
		# We want to show all ticks...
		ax.set_xticks(np.arange(len(col_vals)))
		ax.set_yticks(np.arange(len(row_vals)))
		# ... and label them with the respective list entries
		ax.set_xticklabels(col_vals)
		ax.set_yticklabels(row_vals)
	
		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				 rotation_mode="anchor", fontsize=20)
		
		plt.setp(ax.get_yticklabels(), rotation=0, fontsize=20)
	
		# Loop over data dimensions and create text annotations.
		for i in range(len(row_vals)):
			for j in range(len(col_vals)):
				ax.text(j, i, "{:.1f}".format(val_matrix[i, j]),
						ha="center", va="center", color="w", fontsize=20)
	
		# ax.set_title(f"{metric} for topk={top_k}" if title is None else title)
		ax.set_xlabel("Number of anchor entities", fontsize=20)
		ax.set_ylabel("Number of anchor   mentions", fontsize=20)
		fig.tight_layout()
		if fname is None:
			plt.savefig(f"{curr_res_dir}/{metric}_{top_k}.pdf")
		else:
			plt.savefig(f"{curr_res_dir}/{fname}.pdf")
		plt.close()
	except Exception as e:
		embed()
		raise e


def compute_ment_embeddings(biencoder, mention_tokens_list):
	torch.cuda.empty_cache()
	assert not biencoder.training, "biencoder model should be in eval mode"
	with torch.no_grad():
		bienc_ment_embedding = []
		all_mention_tokens_list_gpu = torch.tensor(mention_tokens_list).to(biencoder.device)
		for ment in all_mention_tokens_list_gpu:
			ment = ment.unsqueeze(0)
			bienc_ment_embedding += [biencoder.encode_input(ment).cpu()]
		
		bienc_ment_embedding = torch.cat(bienc_ment_embedding)
	
	return bienc_ment_embedding
	
	
def compute_bienc_ment_to_ent_matrix(biencoder, mention_tokens_list, candidate_encoding):
	bienc_ment_embedding = compute_ment_embeddings(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
	bienc_all_ment_to_ent_scores = bienc_ment_embedding @ candidate_encoding.T
	return bienc_all_ment_to_ent_scores


def treat_matrix(matrix, method):
	
	if method is None:
		return matrix
	
	
	torch_data = False
	if isinstance(matrix, torch.Tensor):
		matrix = matrix.cpu().numpy()
		torch_data = True
		
	if method == "quantile":
		transformer = preprocessing.QuantileTransformer(random_state=0, output_distribution="normal")
	elif method == "power":
		transformer = preprocessing.PowerTransformer(standardize=False)
	elif method == "power_normalized":
		transformer = preprocessing.PowerTransformer(standardize=True)
	elif method == "remove_large_vals":
		matrix = copy.deepcopy(matrix)
		matrix[matrix > -5] = -5 + (matrix[matrix > -5])/100
		if torch_data:
			matrix = torch.Tensor(matrix)
		return matrix
	# elif method == "eig_val":
	# 	raise NotImplementedError
	else:
		raise NotImplementedError("Method = {} is not supported for transforming the matrix")
	
	trans_matrix = transformer.fit_transform(matrix)
	
	if torch_data:
		trans_matrix = torch.Tensor(trans_matrix)
	
	return trans_matrix
	
	
	

def run(res_dir, data_info, n_seeds, batch_size, arg_dict, biencoder=None):
	try:
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fnames = data_info
		
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fnames["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
		
		
		# LOGGER.info("Loading precomputed ment_to_ent embeds and then computing ment_to_ent scores")
		# with open(data_fnames["crossenc_ment_and_ent_embeds"], "rb") as fin:
		# 	dump_dict = pickle.load(fin)
		#
		# 	all_label_embeds = dump_dict["all_label_embeds"]
		# 	all_input_embeds = dump_dict["all_input_embeds"]
		# 	test_data = dump_dict["test_data"]
		# 	mention_tokens_list = dump_dict["mention_tokens_list"]
		# 	entity_id_list = dump_dict["entity_id_list"]
		# 	entity_tokens_list = dump_dict["entity_tokens_list"]
		#
		# LOGGER.info("Finished loading")
		# crossenc_ment_to_ent_scores = torch.nn.CosineSimilarity(dim=-1)(all_input_embeds, all_label_embeds)
		
		################################################################################################################
		
		total_n_ment, total_n_ent = crossenc_ment_to_ent_scores.shape
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		
		
		####################### RUN MATRIX APPROXIMATION USING CROSSENCODER SCORE MATRIX ###########################
		
		# for norm_method in ["remove_large_vals", "quantile", "power", "power_normalized", None]]:
		for norm_method in [None]:
			run_approx_eval_for_diff_n_anchors(
				ment_to_ent_scores=treat_matrix(
					matrix=crossenc_ment_to_ent_scores,
					method=norm_method
				),
				gt_labels=curr_gt_labels,
				res_dir=f"{res_dir}/crossenc_{norm_method}_nm_{total_n_ment}_ne_{total_n_ent}_s={n_seeds}",
				n_seeds=n_seeds,
				arg_dict=arg_dict
			)
			
		######################## RUN MATRIX APPROXIMATION USING BIENCODER SCORE MATRIX #############################
		
		if biencoder is not None and False:
			biencoder.eval()
			
			LOGGER.info("Loading precomputed entity encodings computed using biencoder")
			# candidate_encoding = np.load(data_fnames["ent_embed_file"])
			complete_entity_tokens_list = torch.LongTensor(np.load(data_fnames["ent_tokens_file"]))
			
			candidate_encoding = compute_label_embeddings(
				biencoder=biencoder,
				labels_tokens_list=complete_entity_tokens_list,
				batch_size=batch_size
			)
			# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
			candidate_encoding = torch.Tensor(candidate_encoding[entity_id_list])

			mention_tokens_list = torch.LongTensor(mention_tokens_list)
			ment_to_ent_scores = compute_bienc_ment_to_ent_matrix(
				biencoder=biencoder,
				mention_tokens_list=mention_tokens_list,
				candidate_encoding=candidate_encoding
			)

			# for norm_method in ["remove_large_vals", "quantile", "power", "power_normalized", None]]:
			for norm_method in [None]:
				run_approx_eval_for_diff_n_anchors(
					ment_to_ent_scores=treat_matrix(
						matrix=ment_to_ent_scores,
						method=norm_method
					),
					gt_labels=curr_gt_labels,
					res_dir=f"{res_dir}/bienc_{norm_method}_nm_{total_n_ment}_ne_{total_n_ent}_s={n_seeds}",
					n_seeds=n_seeds,
					arg_dict=arg_dict
				)
		
		LOGGER.info("Done")
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def main():
	
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	
	parser.add_argument("--bi_model_file", type=str, default="", help="File for biencoder ckpt")
	parser.add_argument("--res_dir", type=str, required=True, help="Res dir with score matrices, and to save results")
	parser.add_argument("--n_seeds", type=int, default=100, help="Number of seeds to run")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions in precomputed mention-entity score matrix")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size to use with biencoder")

	args = parser.parse_args()
	data_name = args.data_name
	
	bi_model_file = args.bi_model_file
	res_dir = args.res_dir
	n_seeds = args.n_seeds
	n_ment = args.n_ment
	batch_size = args.batch_size
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	
	if bi_model_file != "" and bi_model_file.endswith(".json"):
		with open(bi_model_file, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
	elif bi_model_file != "":
		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	else:
		biencoder = None
		
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		res_dir=f"{res_dir}/{data_name}/CUR",
		data_info=(data_name, DATASETS[data_name]),
		n_seeds=n_seeds,
		batch_size=batch_size,
		biencoder=biencoder,
		arg_dict=args.__dict__
	)


if __name__ == "__main__":
	main()

