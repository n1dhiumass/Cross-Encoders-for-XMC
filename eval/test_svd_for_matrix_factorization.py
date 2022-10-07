import os
import sys
import time
import json
import pickle
import logging
import argparse
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def sparsify_mat(dense_mat, n_sparse_per_row, seed):
	
	try:
		n_rows, n_cols = dense_mat.shape
		
		rng = np.random.default_rng(seed=seed)
		all_sampled_sparse_cols = []
		all_sampled_sparse_vals = []
		for row_iter in range(n_rows):
			
			sampled_sparse_cols = rng.choice(n_cols, size=n_sparse_per_row, replace=False)
			all_sampled_sparse_cols += [sampled_sparse_cols]
			all_sampled_sparse_vals += [dense_mat[row_iter, sampled_sparse_cols]]
			
		data = np.array(all_sampled_sparse_vals).reshape(-1)
		sparse_rows = np.array([[row_iter]*n_sparse_per_row for row_iter in range(n_rows)]).reshape(-1)
		sparse_cols = np.array(all_sampled_sparse_cols).reshape(-1)

		
		sparse_mat = csr_matrix((data, (sparse_rows, sparse_cols)), shape=(n_rows, n_cols))
		
		return sparse_mat
	except Exception as e:
		embed()
		raise e



def compute_svd_of_sparse_mat(res_dir, dense_mat, n_sparse_per_row, rank, seed):
	
	n_rows, n_cols = dense_mat.shape
	
	sparse_mat = sparsify_mat(dense_mat=dense_mat, n_sparse_per_row=n_sparse_per_row, seed=seed)
	nnz = sparse_mat.nnz
	LOGGER.info("")
	
	t1 = time.time()
	# U, S, VT = svds(sparse_mat, random_state=seed)
	U, S, VT = svds(sparse_mat, k=rank)
	t2 = time.time()
	
	# V = VT.T
	approx = U @ np.diag(S) @ VT
	frob_error = float(np.linalg.norm(approx - dense_mat, ord="fro"))
	nuc_error = float(np.linalg.norm(approx - dense_mat, ord="nuc"))
	inf_error = float(np.linalg.norm(approx - dense_mat, ord=np.inf))
	
	# corr, _ = pearsonr(approx.reshape(-1), dense_mat.reshape(-1))
	
	plot_score_distribution(
		out_dir=f"{res_dir}/plots/sparsity={n_sparse_per_row}_rank={rank}",
		score_matrices={
			"approx": approx,
			"exact": dense_mat
		}
	)
	result = {
		"error_frob": frob_error,
		"error_nuc": nuc_error,
		"error_inf": inf_error,
		# "pearson_corr": corr,
		# "time": t2-t1,
	}
	
	LOGGER.info(f"Time taken for SVD of {n_rows, n_cols} matrix with {nnz}({nnz/(n_rows*n_cols)}) entries = {t2-t1}")
	LOGGER.info(f"Result = \n{json.dumps(result, indent=4)}")
	# embed()
	
	return  result
	
	
def plot_score_distribution(out_dir, score_matrices, bins=200, num_rows=100):
	
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
		
		out_filename = f"{out_dir}/score_dist_n={num_rows}_scatter_combined.png"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename)
		
		plt.close()
		
		
	except Exception as e:
		embed()
		raise e


def run(base_res_dir, data_info, seed, misc, arg_dict):
	
	try:
		
		
		data_name, data_fname = data_info
	
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores.cpu().numpy()
			crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores - np.mean(crossenc_ment_to_ent_scores)
		
		n_rows, n_cols = crossenc_ment_to_ent_scores.shape
		
		res_dir = f"{base_res_dir}/svd_n_rows={n_rows}_s={seed}_{misc}"
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		
		n_sparse_per_row_vals = [10, 50, 100, 200, 500, 1000, 5000, 10000, n_cols]
		rank_vals = [10, 50, 99, 500, 1000, 2000]
		
		n_sparse_per_row_vals = [100, 500, 1000, 5000, n_cols]
		rank_vals = [10, 50, 99, 500, 1000, 2000]
		
		rank_vals = [v for v in rank_vals if v < n_rows]
		eval_res = {}
		for n_sparse_per_row, rank in itertools.product(n_sparse_per_row_vals, rank_vals):
			res = compute_svd_of_sparse_mat(
				res_dir=res_dir,
				dense_mat=crossenc_ment_to_ent_scores,
				n_sparse_per_row=n_sparse_per_row,
				rank=rank,
				seed=seed
			)
			eval_res[f"sparsity={n_sparse_per_row}_rank={rank}"] = res
		
		
		res_fname = f"{res_dir}/res.json"
		eval_res["other_args"] = {
			"n_sparse_per_row_vals": n_sparse_per_row_vals,
			"rank_vals": rank_vals,
		}
		eval_res["other_args"].update(arg_dict)
		
		Path(os.path.dirname(res_fname)).mkdir(exist_ok=True, parents=True)
		with open(res_fname, "w") as fout:
			json.dump(obj=eval_res, fp=fout, indent=4)


	
	except Exception as e:
		embed()
		raise e
	

	
	
def main():
	
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Test SVD as a potential replacement for CUR matrix factorization')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	
	parser.add_argument("--res_dir", type=str, required=True, help="Res dir with score matrices, and to save results")
	parser.add_argument("--n_ment", type=int, required=True, help="Number of mentions in precomputed mention-entity score matrix")
	# parser.add_argument("--rank", type=int,  required=True, help="Rank of SVD approx of matrix")
	# parser.add_argument("--n_sparse_per_row", type=int,  required=True, help="Number of entries in each row of sparse version of the dense matrix")
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument("--plot_only", type=int, default=0, choices=[0,1], help="1 to only plot results, 0 to run exp and then plot results")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix")
	
	
	

	args = parser.parse_args()
	data_name = args.data_name
	
	res_dir = args.res_dir
	n_ment = args.n_ment
	# rank = args.rank
	# n_sparse_per_row = args.n_sparse_per_row
	seed = args.seed
	plot_only = bool(args.plot_only)
	misc = args.misc
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	

	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		base_res_dir=f"{res_dir}/{data_name}/SVD_Exps",
		data_info=(data_name, DATASETS[data_name]),
		seed=seed,
		misc=misc,
		arg_dict=args.__dict__
	)
 

if __name__ == "__main__":
	main()

