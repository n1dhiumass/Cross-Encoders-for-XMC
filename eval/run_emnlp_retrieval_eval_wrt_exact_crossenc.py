import os
import sys
import json
import torch
import pickle
import logging
import argparse
import wandb
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi

from eval.eval_utils import compute_label_embeddings, compute_input_embeddings, compute_overlap
from eval.matrix_approx_zeshel import CURApprox, plot_heat_map
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS
from eval.nsw_eval_zeshel import compute_ent_embeds_w_tfidf, compute_ment_embeds_w_tfidf
from utils.data_process import read_ent_link_data,load_entities
from eval.nsw_eval_zeshel import compute_ent_embeds_w_tfidf, search_nsw_graph
from eval.eval_utils import compute_label_embeddings, compute_input_embeddings
from models.nearest_nbr import build_flat_or_ivff_index, HNSWWrapper


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
	if torch.is_tensor(indices):
		indices, scores = torch.cat(indices), torch.cat(scores)
		indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	else:
		indices, scores = np.concatenate(indices), np.concatenate(scores)
		
	return {"indices":indices, "scores":scores}


def eval_w_graph_search(
		topk_vals,
		topk_retvr_vals,
		mentions_data,
		entity_file,
		beamsize,
		max_nbrs,
		nsw_metric,
		embed_type,
		biencoder,
		all_ment_to_ent_scores,
		entry_method,
		tokenized_entities,
		tokenized_mentions,
):
	"""
	Find nearest neighbors using cross-encoder model by searching a graph
	:param embed_type: Entity embedding type to retrieve first set of hard negatives
	:param mentions_data: List of mentions along with their contexts.
	:param entity_file: File containing entity information
	:param biencoder: Biencoder model
	:param topk: Number of negatives to return after re-ranking
	:param beamsize: Beam size to use for NSW search
	:param max_nbrs: Max nbr parameter to use when building NSW graph
	:param topk_retvr: Upper limit on number of reranker calls allowed during NSW search
	:return:
	"""
	try:
		n_ments, n_ents = len(tokenized_mentions), len(tokenized_entities)
		
		if embed_type == "bienc":
			LOGGER.info("Computing entity embeddings using bienc")
			ent_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=50)
			ent_embeds = ent_embeds.cpu().detach().numpy()
			
			LOGGER.info("Computing mention embeddings using bienc")
			ment_embeds = compute_input_embeddings(biencoder=biencoder, input_tokens_list=tokenized_mentions, batch_size=50)
			ment_embeds = ment_embeds.cpu().detach().numpy()
			
		elif embed_type == "tfidf":
			mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
						for ment_dict in mentions_data]

			LOGGER.info(f"Embedding {len(mentions_data)} mentions using method = tfidf")
			
			ent_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
			ment_embeds = compute_ment_embeds_w_tfidf(entity_file=entity_file, mentions=mentions)
		
		else:
			raise NotImplementedError(f"Embed type = {embed_type} not supported")
		
		LOGGER.info(f"Finding {beamsize} entry points in the graph using method = {entry_method}")
		if entry_method == "random":
			rng = np.random.default_rng(seed=0)
			temp_init_ents = sorted(rng.choice(n_ents, size=beamsize, replace=False))
			init_ents = [temp_init_ents for _ in range(n_ments)]
		elif entry_method in ["bienc", "tfidf"]:
			nnbr_index = build_flat_or_ivff_index(embeds=ent_embeds, force_exact_search=True)
			_, nn_idxs = nnbr_index.search(ment_embeds, beamsize)
			
			init_ents  = nn_idxs
		else:
			raise NotImplementedError(f"Entry method = {entry_method} not supported")
		
		LOGGER.info(f"Build an NSW graph and searching the graph using scores from re-ranker (crossencoder) model {entity_file}")
		
		######################################### BUILD NSW GRAPH ######################################################
		
		LOGGER.info(f"Building an NSW index over {n_ents} entities with embed shape {ent_embeds.shape} with max_nbrs={max_nbrs}")
		dim = ent_embeds.shape[1]
		index = HNSWWrapper(dim=dim, data=ent_embeds, max_nbrs=max_nbrs, metric=nsw_metric)
		
		LOGGER.info("Extracting lowest level NSW graph from index")
		# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
		nsw_graph = index.get_nsw_graph_at_level(level=1)
		
		################################################################################################################
		
		######################### START NSW SEARCH USING CROSS-ENCODER MODEL ###########################################
		
		ans = {}
		LOGGER.info("Now running graph search")
		for top_k, topk_retvr in tqdm(itertools.product(topk_vals, topk_retvr_vals), total=len(topk_vals)*len(topk_retvr_vals)):
			if topk_retvr < 0: continue
			if top_k > topk_retvr: continue
			ans[f"top_k={top_k}~top_k_retvr={topk_retvr}"] = graph_search_helper(
				topk=top_k,
				topk_retvr=topk_retvr,
				n_ments=n_ments,
				all_ment_to_ent_scores=all_ment_to_ent_scores,
				nsw_graph=nsw_graph,
				beamsize=beamsize,
				init_ents=init_ents
			)
			
		return ans
		
	except Exception as e:
		LOGGER.info("Exception in get_negs_w_nsw_search {e}")
		embed()
		raise e
	


def graph_search_helper(topk, topk_retvr, n_ments, all_ment_to_ent_scores, nsw_graph, beamsize, init_ents):
	
	graph_topk_preds = []
	exact_topk_preds = []
	for ment_idx in tqdm(range(n_ments), position=0, leave=True, total=n_ments):
		
		# Get top-k indices from exact matrix
		curr_m2e_scores = all_ment_to_ent_scores[ment_idx]
		top_k_scores, top_k_indices = curr_m2e_scores.topk(topk)
		exact_topk_preds += [(top_k_indices.unsqueeze(0), top_k_scores.unsqueeze(0))]
		
		# Find top-k entities under given compute budget
		graph_topk_scores , graph_topk_ents, graph_curr_num_score_comps = search_nsw_graph(
			nsw_graph=nsw_graph,
			entity_scores=curr_m2e_scores,
			approx_entity_scores_and_masked_nodes=(None,{}),
			topk=topk,
			arg_beamsize=beamsize,
			init_ents=init_ents[ment_idx],
			comp_budget=topk_retvr,
			exit_at_local_minima_arg=False,
			pad_results=True
		)
		
		assert len(graph_topk_ents) > 0, f"No entity found in nsw search for ment_id = {ment_idx}, {entity_file}"
		
		graph_topk_ents = graph_topk_ents[:topk]
		graph_topk_scores = graph_topk_scores[:topk]
		assert len(graph_topk_ents) == topk, f"len(graph_topk_ents) = {len(graph_topk_ents)} != topk = {topk} "
		
		graph_topk_preds += [(graph_topk_ents.reshape(1, -1), graph_topk_scores.reshape(1, -1))]
	
	exact_topk_preds = _get_indices_scores(exact_topk_preds)
	graph_topk_preds = _get_indices_scores(graph_topk_preds)
	
	# Eval retrieved topk against exact topk
	exact_vs_reranked_approx_retvr = compute_overlap(
		indices_list1=exact_topk_preds["indices"],
		indices_list2=graph_topk_preds["indices"],
	)
	new_exact_vs_reranked_approx_retvr = {}
	for _metric in exact_vs_reranked_approx_retvr:
		new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
		new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
		new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
		
	res = {f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr}
	
	new_res = {} # Convert two level nested dict to a single level dict
	for res_type in res:
		for metric in res[res_type]:
			new_res[f"{res_type}~{metric}"] = res[res_type][metric]
		
	return new_res


def eval_approx_score_mat_for_all_topk(all_ment_to_ent_scores, approx_ment_to_ent_scores, arg_top_k_vals, top_k_retvr):
	"""
	Takes a ground-truth and approximated mention x entity matrix as input, and eval ranking of entities wrt approximate scores
	and also evaluate the approximation.
	This version evaluates for all give top_k_vals after retrieving max_top_k_vals items and then taking subset of those ranked items
	to compute scote for a given top_k value
	:param all_ment_to_ent_scores:
	:param approx_ment_to_ent_scores
	:param top_k_vals:
	:param top_k_retvr:
	:return:
	"""

	
	try:
		n_ments = all_ment_to_ent_scores.shape[0]
	
		exact_topk_preds = []
		approx_topk_preds = []
		topk_w_approx_retrvr_preds = []
		
		top_k_vals = [top_k for top_k in arg_top_k_vals if top_k <= top_k_retvr] # Ignore top_k values that are larger than top_k_retvr value
		if len(top_k_vals) == 0: # If there are no top_k_vals left then return
			return {}
		
		max_topk = max(top_k_vals)
		
		
	
		for ment_idx in range(n_ments):
	
			curr_ment_scores = all_ment_to_ent_scores[ment_idx]
			approx_curr_ment_scores = approx_ment_to_ent_scores[ment_idx]
	
			# Get top-k indices from exact matrix
			top_k_scores, top_k_indices = curr_ment_scores.topk(max_topk)
	
			# Get top-k indices from approx-matrix
			approx_top_k_scores, approx_top_k_indices = approx_curr_ment_scores.topk(top_k_retvr)
			
			# Re-rank top-k indices from approx-matrix using exact scores from ment_to_ent matrix
			# Scores from given ment_to_ent matrix filled only for entities retrieved by approximate retriever
			temp = torch.zeros(curr_ment_scores.shape) - 99999999999999
			temp[approx_top_k_indices] = curr_ment_scores[approx_top_k_indices]
		
			top_k_w_approx_retrvr_scores, top_k_w_approx_retrvr_indices = temp.topk(max_topk)
			
			exact_topk_preds += [(top_k_indices.unsqueeze(0), top_k_scores.unsqueeze(0))]
			approx_topk_preds += [(approx_top_k_indices.unsqueeze(0), approx_top_k_scores.unsqueeze(0))]
			topk_w_approx_retrvr_preds += [(top_k_w_approx_retrvr_indices.unsqueeze(0), top_k_w_approx_retrvr_scores.unsqueeze(0))]
	
		
		exact_topk_preds = _get_indices_scores(exact_topk_preds)
		approx_topk_preds = _get_indices_scores(approx_topk_preds)
		topk_w_approx_retrvr_preds = _get_indices_scores(topk_w_approx_retrvr_preds)
		
		
		
		res_for_topk = {}
		for top_k in top_k_vals:
			# Eval for each top_k value
			exact_vs_reranked_approx_retvr = compute_overlap(
				indices_list1=exact_topk_preds["indices"][:, :top_k],
				indices_list2=topk_w_approx_retrvr_preds["indices"][:, :top_k],
			)
			new_exact_vs_reranked_approx_retvr = {}
			for _metric in exact_vs_reranked_approx_retvr:
				new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
			
			res = {f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr}
			
			new_res = {} # Convert two level nested dict to a single level dict
			for res_type in res:
				for metric in res[res_type]:
					new_res[f"{res_type}~{metric}"] = res[res_type][metric]
			
		
			res_for_topk[top_k] = new_res
		
		return res_for_topk
	except Exception as e:
		embed()
		raise e


def eval_approx_score_mat(all_ment_to_ent_scores, approx_ment_to_ent_scores, top_k, top_k_retvr):
	"""
	Takes a ground-truth and approximated mention x entity matrix as input, and eval ranking of entities wrt approximate scores
	and also evaluate the approximation
	:param all_ment_to_ent_scores:
	:param approx_ment_to_ent_scores
	:param top_k:
	:param top_k_retvr:
	:return:
	"""

	
	try:
		n_ments = all_ment_to_ent_scores.shape[0]
	
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
		
		exact_vs_reranked_approx_retvr = compute_overlap(
			indices_list1=topk_preds["indices"],
			indices_list2=topk_w_approx_retrvr_preds["indices"],
		)
		new_exact_vs_reranked_approx_retvr = {}
		for _metric in exact_vs_reranked_approx_retvr:
			new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
			new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
			new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
		
		res = {f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr}
		
		new_res = {} # Convert two level nested dict to a single level dict
		for res_type in res:
			for metric in res[res_type]:
				new_res[f"{res_type}~{metric}"] = res[res_type][metric]
		
		return new_res
		
	except Exception as e:
		embed()
		raise e


def run_eval_method(curr_method, test_data_file, train_data_file, fixed_anc_ent_args, bienc_args, cur_args, tfidf_args, graph_args, use_wandb):
	"""
	Evaluate a given method on test data
	:param data_info:
	:param batch_size:
	:param biencoder:
	:return:
	"""
	try:
	
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(test_data_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			test_crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_mention_tokens_list = dump_dict["mention_tokens_list"]
			# test_data = dump_dict["test_data"]
			# entity_id_list = dump_dict["entity_id_list"]
			test_mention_tokens_list = torch.LongTensor(test_mention_tokens_list)
			test_total_n_ment, test_total_n_ent = test_crossenc_ment_to_ent_scores.shape
			test_ment_idxs = dump_dict["ment_idxs"]
			
		with open(train_data_file, "rb") as fin:
				dump_dict = pickle.load(fin)
				train_crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				train_total_n_ment, train_total_n_ent = train_crossenc_ment_to_ent_scores.shape
				
				assert train_total_n_ent == test_total_n_ent, "Train and test entities differ! Use entity_id_list from data dump to resolve this"
			
		
		if use_wandb: wandb.log({"eval_status":0})
		
		
		top_k_vals = [1, 10, 50, 100]
		top_k_retr_vals_base = [1, 10, 50, 100, 200, 500, 1000]
		
		top_k_retr_vals_cur =  top_k_retr_vals_base + [int(k*frac) for k in top_k_retr_vals_base for frac in np.arange(0.1, 1.0, 0.1)]
		
		if "cur" in curr_method or "fixed_anc_ent" in curr_method:
			top_k_retr_vals = top_k_retr_vals_cur
		else:
			top_k_retr_vals  = top_k_retr_vals_base
		top_k_retr_vals  = sorted(list(set(top_k_retr_vals)))
		
		n_ent_anchors_vals_base = [10, 50, 100, 200, 500, 1000, 2000]
		n_ent_anchors_vals = [v for v in n_ent_anchors_vals_base if v < test_total_n_ent] + [test_total_n_ent]
		n_ent_anchors_vals = sorted(list(set(n_ent_anchors_vals + top_k_retr_vals_cur)))
		
		LOGGER.info(f"Computing approximate test mention-to-entity scores using method={curr_method}")
		
		test_approx_ment_to_ent_scores = {}
		# Compute approximate test mention-to-entity scores
		if curr_method == "bienc":
		
			bi_model_file = bienc_args["bi_model_file"]
			batch_size = bienc_args["batch_size"]
			entity_token_file = bienc_args["entity_token_file"]
			
			complete_entity_tokens_list = torch.LongTensor(np.load(entity_token_file))
			try:
				biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file) if bi_model_file != "" else None
			except:
				LOGGER.info("Loading biencoder which was trained while sharing parameters with a cross-encoder.")
				biencoder = CrossEncoderWrapper.load_from_checkpoint(bi_model_file) if bi_model_file != "" else None
			
			biencoder.eval()
		
			label_embeds = compute_label_embeddings(
				biencoder=biencoder,
				labels_tokens_list=complete_entity_tokens_list,
				batch_size=batch_size
			)

			mention_embeds = compute_input_embeddings(
				biencoder=biencoder,
				input_tokens_list=test_mention_tokens_list,
				batch_size=batch_size
			)
			test_bienc_ment_to_ent_scores = mention_embeds @ label_embeds.T
			test_approx_ment_to_ent_scores = {n_ent_anchors:test_bienc_ment_to_ent_scores for n_ent_anchors in n_ent_anchors_vals}
		
		elif curr_method == "cur":
			seed = cur_args["seed"]
			
			rng = np.random.default_rng(seed=seed)
			
			# Read training data, build latent entity embeds, and then approximation for test data
			
			for n_ent_anchors in n_ent_anchors_vals:
				
				anchor_ent_idxs = sorted(rng.choice(test_total_n_ent, size=n_ent_anchors, replace=False))
				
				cols = train_crossenc_ment_to_ent_scores[:, anchor_ent_idxs]
				approx = CURApprox(row_idxs=np.arange(train_total_n_ment), col_idxs=anchor_ent_idxs, rows=train_crossenc_ment_to_ent_scores, cols=cols, approx_preference="rows")
				
				test_crossenc_ment_to_anchor_ents = test_crossenc_ment_to_ent_scores[:, anchor_ent_idxs]
				test_cur_ment_to_ent_scores = approx.get_complete_row(sparse_rows=test_crossenc_ment_to_anchor_ents)
				
				test_approx_ment_to_ent_scores[n_ent_anchors] = test_cur_ment_to_ent_scores
		
		elif curr_method == "fixed_anc_ent":
			
			e2e_fname = fixed_anc_ent_args["e2e_fname"]
			n_fixed_anc_ent = fixed_anc_ent_args["n_fixed_anc_ent"]
			
			if not os.path.isfile(e2e_fname):
				LOGGER.info(f"File {e2e_fname} not found")
			
			with open(e2e_fname, "rb") as fin:
				dump_dict = pickle.load(fin)
				full_ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : Number of entities x Number of anchors
				full_anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of anchors
		
			anchor_ent_idxs = full_anchor_ents[:n_fixed_anc_ent] # Select first n_anc_ent entities as anchor entities
			ent_embeds = full_ent_embeds[:, :n_fixed_anc_ent] # Take scores with only n_anc_ent anchor entities
			
			# Get mention scores with fixed anchor entities
			mention_embeds = test_crossenc_ment_to_ent_scores[:, anchor_ent_idxs]
			
			temp_ans = mention_embeds @ ent_embeds.T
			test_approx_ment_to_ent_scores = {dummy_n_anc_ent:temp_ans for dummy_n_anc_ent in n_ent_anchors_vals}
				
		elif curr_method == "fixed_anc_ent_cur":
			
			e2e_fname = fixed_anc_ent_args["e2e_fname"]
			n_fixed_anc_ent = fixed_anc_ent_args["n_fixed_anc_ent"]
			
			if not os.path.isfile(e2e_fname):
				LOGGER.info(f"File {e2e_fname} not found")
			
			with open(e2e_fname, "rb") as fin:
				dump_dict = pickle.load(fin)
				full_ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : (Number of entities, Number of fixed anchors ents)
				fixed_anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of fixed anchor entities
				n_ents = full_ent_embeds.shape[0]
				
				R = full_ent_embeds[:, :n_fixed_anc_ent].T # shape : (n_fixed_anchor_ents, n_ents)
				
			rng = np.random.default_rng(seed=0)
			
			for n_anc_ent in n_ent_anchors_vals:
				
				# Choose indices of anchor entities
				anchor_ent_idxs = sorted(rng.choice(n_ents, size=n_anc_ent, replace=False))
				
				intersect_mat = R[:, anchor_ent_idxs] # shape: (n_fixed_anchor_ents, n_anc_ent)
				U = torch.tensor(np.linalg.pinv(intersect_mat))  # shape: (n_anc_ent, n_fixed_anchor_ents)
				UR = U @ R # shape: (n_anc_ent, n_ents)
				
				# Score of mentions w/ anchor entities,
				mention_embeds = test_crossenc_ment_to_ent_scores[:, anchor_ent_idxs] # shape: (n_ments, n_anc_ent)
				
				# (n_ments, n_ents) = (n_ments, n_anc_ent) x (n_anc_ent, n_ents)
				test_approx_ment_to_ent_scores[n_anc_ent] = mention_embeds @ UR
		
		elif curr_method == "tfidf":
			
			mention_file = tfidf_args["mention_file"]
			entity_file = tfidf_args["entity_file"]
			
			LOGGER.info(f"Reading data from {mention_file}, {entity_file}")
			mentions_data, entity_data  = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)

			############################# GET MENTION AND ENTITY EMBEDDINGS ############################################
			mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
						for ment_dict in mentions_data]

			LOGGER.info(f"Embedding {len(mentions_data)} mentions using method = tfidf")
			mention_embeds = compute_ment_embeds_w_tfidf(
				entity_file=entity_file,
				mentions=mentions
			)
			
			mention_embeds = mention_embeds[test_ment_idxs]

			LOGGER.info(f"Embedding entities using method = tfidf")
			label_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
			
			test_tfidf_ment_to_ent_scores = torch.tensor(np.matmul(mention_embeds,np.transpose(label_embeds)))
			
			test_approx_ment_to_ent_scores = {n_ent_anchors:test_tfidf_ment_to_ent_scores for n_ent_anchors in n_ent_anchors_vals}
		
		elif curr_method == "bm25":
			
			mention_file = tfidf_args["mention_file"]
			entity_file = tfidf_args["entity_file"]
			
			LOGGER.info(f"Reading data from {mention_file}, {entity_file}")
			mentions_data, entity_data  = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)

			############################# TRAIN BM25 on ENTITY DATA ############################################
			mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
						for ment_dict in mentions_data]
			
			(title2id,
			id2title,
			id2text,
			kb_id2local_id) = entity_data
			
			LOGGER.info("\t\tTraining BM25 on entity descriptions")
			### Build a list of entity description and train a vectorizer
			corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
			
			tokenized_corpus = [doc.split(" ") for doc in corpus]
			
			bm25 = BM25Okapi(tokenized_corpus)
			
			############################# SCORE ENTITIES USING BM25 ############################################
			LOGGER.info("\t\tScoring all documents using BM25 model")
		
			selected_ments = [mentions[idx] for idx in test_ment_idxs]
			doc_scores  = [bm25.get_scores(ment.split(" ")) for ment in tqdm(selected_ments)]
			doc_scores  = torch.tensor(doc_scores)
			
			test_approx_ment_to_ent_scores = {n_ent_anchors:doc_scores for n_ent_anchors in n_ent_anchors_vals}
		
		elif curr_method == "graph":
			test_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
		elif curr_method == "anchor_ents":
			# Implement scoring using entity embeddings from anchoring against some fixed set of entities
			# (instead of mentions) and also embedding mentions against those fixed set of embeddings.
			# FIXME: Was the anchor entity ordering same for for mention embedding and entity embeddding when this method was tried previously?
			raise NotImplementedError(f"Method = {curr_method} not supported")
		else:
			raise NotImplementedError(f"Method = {curr_method} not supported")
		
		if use_wandb: wandb.log({"eval_status":1})
	
		LOGGER.info(f"Evaluating approximate ment_to_ent scores computed using {curr_method}")
	
		# # Find top-k entities for each mention using approximate ment2ent scores
		# eval_res = defaultdict(lambda : defaultdict(dict))
		# tuple_list = list(itertools.product(top_k_retr_vals, top_k_vals, n_ent_anchors_vals))
		# for ctr, (top_k_retvr, top_k, n_ent_anchors) in enumerate(tqdm(tuple_list)):
		# 	if use_wandb: wandb.log({"eval_ctr_frac":ctr/len(tuple_list)})
		#
		# 	if top_k_retvr < top_k: continue
		# 	if top_k_retvr < 0: continue
		# 	if top_k_retvr > test_total_n_ent: continue
		# 	if n_ent_anchors not in test_approx_ment_to_ent_scores: continue
		#
		# 	# Avoid repeating bienc eval for different n_ent_anchors as it does not make any difference for biencoder model
		# 	if curr_method == "bienc" and n_ent_anchors != n_ent_anchors_vals[0]:
		# 		prev_ans = eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors_vals[0]}"]
		# 		eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors}"] = prev_ans
		# 		continue
		#
		# 	eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors}"] = eval_approx_score_mat(
		# 		all_ment_to_ent_scores=test_crossenc_ment_to_ent_scores,
		# 		approx_ment_to_ent_scores=test_approx_ment_to_ent_scores[n_ent_anchors],
		# 		top_k=top_k,
		# 		top_k_retvr=top_k_retvr
		# 	)
		
		eval_res = defaultdict(lambda : defaultdict(dict))
		tuple_list = list(itertools.product(top_k_retr_vals, n_ent_anchors_vals))
		for ctr, (top_k_retvr, n_ent_anchors) in enumerate(tqdm(tuple_list)):
			if use_wandb: wandb.log({"eval_ctr_frac":ctr/len(tuple_list)})


			if top_k_retvr < 0: continue
			if top_k_retvr > test_total_n_ent: continue
			if n_ent_anchors not in test_approx_ment_to_ent_scores: continue

			# Avoid repeating bienc eval for different n_ent_anchors as it does not make any difference for biencoder model
			if curr_method in ["bienc", "tfidf", "bm25", "fixed_anc_ent", "graph"] and n_ent_anchors != n_ent_anchors_vals[0]:
				for top_k in top_k_vals:
					if top_k > top_k_retvr: continue # Ignore top_k values that are larger than top_k_retvr

					prev_ans = eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors_vals[0]}"]
					eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors}"] = prev_ans

				continue

			if curr_method in ["graph"] and top_k_retvr != top_k_retr_vals[0]:
				continue
				
			if curr_method == "graph":
				mention_file = graph_args["mention_file"]
				entity_file = graph_args["entity_file"]
				bi_model_file = graph_args["bi_model_file"]
				entity_token_file = graph_args["entity_token_file"]
			
				tokenized_entities = torch.LongTensor(np.load(entity_token_file))
			
				LOGGER.info(f"Reading data from {mention_file}, {entity_file}")
				test_mentions_data, _ = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
				test_mentions_data = [test_mentions_data[idx] for idx in test_ment_idxs]
				
				biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
				biencoder.eval()
				
				
				eval_res_for_curr_topk_n_topk_retvr_vals = eval_w_graph_search(
					all_ment_to_ent_scores=test_crossenc_ment_to_ent_scores,
					mentions_data=test_mentions_data,
					entity_file=entity_file,
					topk_vals=top_k_vals,
					topk_retvr_vals=top_k_retr_vals,
					embed_type=graph_args["embed_type"],
					beamsize=graph_args["beamsize"],
					max_nbrs=graph_args["max_nbrs"],
					nsw_metric="l2",
					biencoder=biencoder,
					entry_method=graph_args["entry_method"],
					tokenized_entities=tokenized_entities,
					tokenized_mentions=test_mention_tokens_list
				)
				for top_k, inner_top_k_retvr in itertools.product(top_k_vals, top_k_retr_vals):
					if inner_top_k_retvr < 0: continue
					if top_k > inner_top_k_retvr: continue
					eval_res[f"top_k={top_k}"][f"k_retvr={inner_top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors}"] = eval_res_for_curr_topk_n_topk_retvr_vals[f"top_k={top_k}~top_k_retvr={inner_top_k_retvr}"]
					
			else:
				eval_for_all_top_k_vals = eval_approx_score_mat_for_all_topk(
					all_ment_to_ent_scores=test_crossenc_ment_to_ent_scores,
					approx_ment_to_ent_scores=test_approx_ment_to_ent_scores[n_ent_anchors],
					arg_top_k_vals=top_k_vals,
					top_k_retvr=top_k_retvr
				)
				for top_k in top_k_vals:
					if top_k > top_k_retvr: continue # Ignore top_k values that are larger than top_k_retvr
	
					eval_res[f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={train_total_n_ment}_anc_n_e={n_ent_anchors}"] = eval_for_all_top_k_vals[top_k]
			
		
		if use_wandb: wandb.log({"eval_status":2})
		
		retrieval_params = {
			"top_k_retr_vals": top_k_retr_vals,
			"top_k_vals": top_k_vals,
			"n_ent_anchors_vals": n_ent_anchors_vals
		}
		return eval_res, retrieval_params
	
	except Exception as e:
		embed()
		raise e
	

def run(data_info, res_dir, eval_method, test_data_file, bi_model_file, batch_size, train_data_file, e2e_fname,
		n_fixed_anc_ent, mention_file, entity_file, embed_type, max_nbrs, beamsize, entry_method,
		n_seeds, misc, use_wandb, arg_dict):
	
	data_name, data_fname = data_info
	
	entity_token_file = data_fname["ent_tokens_file"]
	eval_res = {}
	
	assert eval_method == "cur" or n_seeds == 1, f"n_seed = {n_seeds} only allowed for eval_method = cur "
	assert eval_method != "bienc" or n_seeds == 1, f"n_seed = {n_seeds} not allowed for eval_method = bienc "
	
	if use_wandb:
		os.environ["WANDB_API_KEY"] = "6ae7d53ecce3f7d824317087c3973ebd50e29bb3"
		wandb.init(
			project="8_CUR_EMNLP_Eval",
			dir="../../results/_8_CUR_EMNLP_Eval",
			config=arg_dict,
			mode="online" if use_wandb else "disabled"
		)
	
	retvr_params = {}
	for seed in range(n_seeds):
		if use_wandb: wandb.log({"seed_ctr":seed})
		curr_res, retvr_params = run_eval_method(
			curr_method=eval_method,
			test_data_file=test_data_file,
			train_data_file=train_data_file,
			bienc_args={
				"bi_model_file":bi_model_file,
				"batch_size":batch_size,
				"entity_token_file":entity_token_file,
			},
			cur_args={
				"seed":seed,
			},
			fixed_anc_ent_args={
				"e2e_fname": e2e_fname,
				"n_fixed_anc_ent" : n_fixed_anc_ent,
			},
			tfidf_args={
				"mention_file": mention_file,
				"entity_file": entity_file,
			},
			graph_args={
				"mention_file":mention_file,
				"entity_file":entity_file,
				"bi_model_file":bi_model_file,
				"embed_type":embed_type,
				"max_nbrs":max_nbrs,
				"beamsize":beamsize,
				"entry_method":entry_method,
				"entity_token_file":entity_token_file,
			},
			use_wandb=use_wandb,
		)
		eval_res[f"seed={seed}"] = curr_res
	
	if use_wandb: wandb.log({"eval_status":4})
	eval_res["other_args"] = arg_dict
	eval_res["other_args"]["retriever_params"] = retvr_params
	
	res_file = f"{res_dir}/method={eval_method}_{misc}.json"
	Path(os.path.dirname(res_file)).mkdir(exist_ok=True, parents=True)
	with open(res_file, "w") as fout:
		json.dump(eval_res, fout, indent=4)

	
	if use_wandb: wandb.log({"eval_status":5})
	
	# TODO: Read res_file and average results for all seeds if eval_method == cur
	
	if use_wandb: wandb.log({"eval_status":6})


def main():
	
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run eval for various retrieval methods wrt exact crossencoder scores. This evaluation does not use ground-truth entity information into account')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	parser.add_argument("--eval_method", type=str, choices=["cur", "bienc", "fixed_anc_ent", "fixed_anc_ent_cur", "tfidf", "bm25", "graph"], help="Eval method")
	parser.add_argument("--res_dir", type=str, required=True, help="Result directory")
	parser.add_argument("--test_data_file", type=str, required=True, help="Test data file")
	
	
	
	# Params for eval_method = cur
	parser.add_argument("--train_data_file", type=str, default="", help="Training data file. Used for method=cur")
	parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds to run (in case there is any source of randomness eg choosing anchor entities for CUR")
	
	# Params for eval_method = bienc
	parser.add_argument("--bi_model_file", type=str, default="", help="File for biencoder ckpt. Used for method=bienc")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size to use with biencoder. Used for method=bienc")
	
	# Params for eval_method = fixed_anc_ent or fixed_anc_ent_cur
	parser.add_argument("--e2e_fname", type=str, default="", help="File w/ entity2entity scores. Used for method=fixed_anc_ent")
	parser.add_argument("--n_fixed_anc_ent", type=int, default=0, help="Number of fixed anchor entities to use. Used for method=fixed_anc_ent")
	
	# Params for eval_method = tfidf or BM25
	parser.add_argument("--mention_file", type=str, default="", help="File containing raw mention data. Used for method=tfidf or BM25")
	parser.add_argument("--entity_file", type=str, default="", help="File containing raw entity data. Used for method=tfidf or BM25")
	
	# Params for eval_method = graph
	parser.add_argument("--embed_type", type=str, default="", help="Used for method=graph")
	parser.add_argument("--max_nbrs", type=int, default=5, help="Used for method=graph")
	parser.add_argument("--beamsize", type=int, default=0, help="Used for method=graph")
	parser.add_argument("--entry_method", type=str, default="", help="Used for method=graph")
	
	# Some extra params for more control on how eval is done
	parser.add_argument("--mode", type=str, choices=["eval", "plot", "eval_n_plot"], default="eval", help="To run in eval mode or just plotting or both")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix")
	parser.add_argument("--use_wandb", type=int, default=0, choices=[0, 1], help="1 to enable wandb and 0 to disable it ")

						
	args = parser.parse_args()
	data_name = args.data_name
	eval_method = args.eval_method
	res_dir = args.res_dir
	test_data_file = args.test_data_file
	
	train_data_file = args.train_data_file
	n_seeds = args.n_seeds
	
	e2e_fname = args.e2e_fname
	n_fixed_anc_ent = args.n_fixed_anc_ent

	# Params for eval_method = tfidf or BM25
	entity_file = args.entity_file
	mention_file = args.mention_file
	
	# Params for eval_method = graph
	embed_type = args.embed_type
	max_nbrs = args.max_nbrs
	beamsize = args.beamsize
	entry_method = args.entry_method

	
	bi_model_file = args.bi_model_file
	batch_size = args.batch_size
	
	mode = args.mode
	misc = args.misc
	use_wandb = bool(args.use_wandb)
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		eval_method=eval_method,
		res_dir=res_dir,
		train_data_file=train_data_file,
		test_data_file=test_data_file,
		e2e_fname=e2e_fname,
		n_fixed_anc_ent=n_fixed_anc_ent,
		mention_file=mention_file,
		entity_file=entity_file,
		bi_model_file=bi_model_file,
		data_info=(data_name, DATASETS[data_name]),
		batch_size=batch_size,
		embed_type=embed_type,
		max_nbrs=max_nbrs,
		beamsize=beamsize,
		entry_method=entry_method,
		use_wandb=use_wandb,
		misc=misc,
		n_seeds=n_seeds,
		arg_dict=args.__dict__
	)


if __name__ == "__main__":
	main()

