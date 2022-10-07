import os
import sys
import json
import torch
import types
import wandb
import pickle
import warnings
import logging
import argparse
import itertools
import numpy as np

import faiss
from tqdm import tqdm
from IPython import embed
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from eval.eval_utils import score_topk_preds, compute_overlap, compute_label_embeddings
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
from utils.data_process import load_entities
from models.nearest_nbr import HNSWWrapper, KNNGraph, KNNGraphwEnt2Ent, RandomGraph, build_flat_or_ivff_index, NSWGraphwEnt2Ent
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


NUM_ENTS = {"world_of_warcraft" : 27677,
			"starwars" : 87056,
			"pro_wrestling" : 10133,
			"military" : 104520,
			"final_fantasy" : 14044,
			"fallout" : 16992,
			"american_football" : 31929,
			"doctor_who" : 40281}


def _get_indices_scores(topk_preds):
	"""
	Convert a list of indices,scores tuple to two list by concatenating all indices and all scores together.
	:param topk_preds: List of indices,scores tuple
	:return: dict with two keys "indices" and "scores" mapping to lists
	"""
	if len(topk_preds) == 0:
		return {"indices":[], "scores":[]}
	indices, scores = zip(*topk_preds)
	if torch.is_tensor(indices[0]):
		indices, scores = torch.stack(indices), torch.stack(scores)
		indices, scores = indices.cpu().numpy(), scores.cpu().numpy()
	else:
		indices, scores = np.stack(indices), np.stack(scores)
	
	return {"indices":indices, "scores":scores}


def _get_topk(curr_node_n_score_tuples, topk):
	"""
	Get topk nodes and their scores, sorted from highest to lowest score
	:param curr_node_n_score_tuples: List of (node_id, score) tuples
	:param topk:
	:return:
	"""
	
	curr_node_n_score_tuples = sorted(curr_node_n_score_tuples, key=lambda x:x[1], reverse=True)
	return curr_node_n_score_tuples[:topk]


def compute_ment_embeds_w_tfidf(entity_file, mentions):
	"""
	Trains a tfidf vectorizer using entity file and then vectorizes mentions using trained tfidf vectorizer
	:param entity_file: File containing entity information
	:param mentions: List of mention strings.
	:return: TF_IDF embeddings of mentions
	"""
	LOGGER.info("\t\tLoading entity descriptions")
	# Read entity descriptions and embed using BM25/tf-idf
	(title2id,
	id2title,
	id2text,
	kb_id2local_id) = load_entities(entity_file=entity_file)
	
	LOGGER.info("\t\tTraining vectorizer on entity descriptions")
	### Build a list of entity description and train a vectorizer
	corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
	vectorizer = TfidfVectorizer(dtype=np.float32)
	vectorizer.fit(corpus)
	
	### Embed all entities usign tfidf vectorizer
	LOGGER.info("\t\tTransforming mentions to sparse vectors")
	
	ment_embeds = vectorizer.transform(mentions)

	return np.asarray(ment_embeds.todense())
	
	
def compute_ment_embeds_w_bienc(biencoder, mention_tokens_list):
	"""
	Embed mentions using biencoder model
	:param biencoder: Biencoder model for embedding mentions
	:param mention_tokens_list: List of tokenized mentions
	:return: torch tensor containing mention embeddings
	"""
	with torch.no_grad():
		assert not biencoder.training, "Biencoder should be in eval mode"
		torch.cuda.empty_cache()
		bienc_ment_embedding = []
		all_mention_tokens_list_gpu = torch.tensor(mention_tokens_list).to(biencoder.device)
		for ment in all_mention_tokens_list_gpu:
			ment = ment.unsqueeze(0)
			bienc_ment_embedding += [biencoder.encode_input(ment)]
		
		bienc_ment_embedding = torch.cat(bienc_ment_embedding)
	
	return bienc_ment_embedding
	

def compute_ment_embeds(embed_type, entity_file, mentions, biencoder, mention_tokens_list):
	"""
	Computes  mention embeddings with given method
	:param embed_type: Method to use for computing mention embeddings
	:param entity_file: File containing entity information
	:param mentions: List of mention strings.
	:param biencoder: Biencoder model for embedding mentions
	:param mention_tokens_list: List of tokenized mentions
	:return: Array containing mention embeddings
	"""
	if embed_type == "tfidf":
		ment_embeds = compute_ment_embeds_w_tfidf(
			entity_file=entity_file,
			mentions=mentions
		)
		return ment_embeds
	elif embed_type == "bienc":
		LOGGER.info("\t\tComputing mention embedding using biencoder")
		ment_embeds = compute_ment_embeds_w_bienc(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
		ment_embeds = ment_embeds.cpu().detach().numpy()
		return ment_embeds
	elif embed_type == "random":
		return None
	else:
		raise NotImplementedError(f"Method = {embed_type} not supported for computing mention embeddings")

	
def compute_ment_to_ent_matrix_w_bienc(biencoder, mention_tokens_list, ent_embeddings):
	"""
	Compute mention embeddings and then uses them with ent_embeddings to compute mention-entity score matrix
	:param biencoder: Model to use for embedding mentions
	:param mention_tokens_list: List of tokenized mentions
	:param ent_embeddings: Entity embeddings
	:return: Array of shape (n_ments x n_ents) containing mention-entity scores.
	"""
	assert not biencoder.training, "Biencoder should be in eval mode"
	bienc_ment_embedding = compute_ment_embeds_w_bienc(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
	bienc_ment_embedding = bienc_ment_embedding.to(ent_embeddings.device)
	bienc_all_ment_to_ent_scores = bienc_ment_embedding @ ent_embeddings.T
	return bienc_all_ment_to_ent_scores


def compute_ment_to_ent_matrix_w_tfidf(entity_file, mentions):
	"""
	1. Trains a tfidf vectorizer using entity file
	2. Vectorizes mentions and entities using trained tfidf vectorizer.
	3. Finally computes mention-entity score matrix using dot product of their embeddings
	:param entity_file: File containing entity information
	:param mentions: List of mention strings.
	:return: Array of shape n_ments x n_ents with mention-entity scores
			computed using dot product of mention and entity embeddings
	"""
	LOGGER.info("\t\tLoading entity descriptions")
	# Read entity descriptions and embed using BM25/tf-idf
	(title2id,
	id2title,
	id2text,
	kb_id2local_id) = load_entities(entity_file=entity_file)
	
	LOGGER.info("\t\tTraining vectorizer on entity descriptions")
	### Build a list of entity description and train a vectorizer
	corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
	vectorizer = TfidfVectorizer(dtype=np.float32)
	vectorizer.fit(corpus)
	
	### Embed all entities usign tfidf vectorizer
	LOGGER.info("\t\tTransforming mentions to sparse vectors")
	
	ment_embeds = vectorizer.transform(mentions)
	ent_embeds = vectorizer.transform(corpus)
	
	scores = ment_embeds * ent_embeds.T
	return np.asarray(scores.todense())
	

def compute_ent_embeds_w_tfidf(entity_file):
	"""
	Trains a tf-idf vectorizer over entity title and text, vectorizes them, and returns dense tfidf embeddings
	:param entity_file: File containing entity information
	:return:
	"""
	LOGGER.info("Loading entity descriptions")
	# Read entity descriptions and embed using BM25/tf-idf
	(title2id,
	id2title,
	id2text,
	kb_id2local_id) = load_entities(entity_file=entity_file)
	
	LOGGER.info("Training ")
	### Build a list of entity description and train a vectorizer
	corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
	vectorizer = TfidfVectorizer(dtype=np.float32)
	vectorizer.fit(corpus)
	
	### Embed all entities usign tfidf vectorizer
	LOGGER.info("Transforming entity to sparse vectors")
	label_embeds = vectorizer.transform(corpus)
	
	
	return label_embeds.todense()


def compute_ent_embeds(embed_type, biencoder, entity_tokens_file, entity_file):
	"""
	Return entity embeddings as dense numpy arrays
	:param embed_type: Method to use for computing entity embeddings
	:param biencoder: Biencoder model for embedding entities
	:param entity_tokens_file: File containing tokenized entities
	:param entity_file: File containing entity information
	:return: entity embeddings as dense numpy arrays
	"""
	if embed_type == "bienc":
		LOGGER.info("Computing entity encodings computed using biencoder")
		complete_entity_tokens_list = np.load(entity_tokens_file)
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		all_ent_embeddings = compute_label_embeddings(
			biencoder=biencoder,
			labels_tokens_list=complete_entity_tokens_list,
			batch_size=200
		)
		return all_ent_embeddings.cpu().numpy()
	elif embed_type == "tfidf":
		LOGGER.info("Computing entity encodings computed using tfidf")
		return compute_ent_embeds_w_tfidf(entity_file=entity_file)
	elif embed_type == "random":
		return None
	else:
		raise Exception(f"Embed_type = {embed_type} is not supported")
		

def get_index(index_path, max_nbrs, embed_type, graph_metric, entity_file, bienc_ent_embeds, ment_to_ent_scores, e2e_score_filename="", graph_type="nsw", n_long_range_frac=0.2):
	"""
	Loads index from given path if available, else builds HNSW index using entity info and embed_type param
	:param index_path: Path to index to load
	:param graph_type: Type of graph to build eg nsw or knn
	:param max_nbrs: Max nbr parameter in NSW graph
	:param embed_type: Type of entity embedding to use for building index
	:param entity_file: File containing entity information
	:param bienc_ent_embeds: Array containing entity embeddings
	:param ment_to_ent_scores: Array of shape (n_ments x n_ents) containing mention-entity scores.
	:param graph_metric: How to measure similarity/distance b/w vectors when building NSW graph
	:param e2e_score_filename: Parameter useful for knn_e2e graph_type
	:return: Index build or loaded using given parameters
	"""
	if index_path is not None and os.path.isfile(index_path):
		LOGGER.info(f"\n\nLoading precomputed index from {index_path}\n\n")
		index = HNSWWrapper(dim=1, data=[], max_nbrs=max_nbrs, metric=graph_metric)
		index.deserialize_from(index_path)
	else:
		LOGGER.info(f"\n\nBuilding index as index_path = {index_path} and index file exists = {os.path.isfile(index_path) if index_path is not None else None}\n\n")
		if embed_type == "tfidf":
			label_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
		elif embed_type == "bienc" or embed_type == "c-anchor":
			label_embeds = bienc_ent_embeds
		elif embed_type == "anchor":
			if torch.is_tensor(ment_to_ent_scores):
				ment_to_ent_scores = ment_to_ent_scores.cpu().detach().numpy()
			label_embeds = np.ascontiguousarray(np.transpose(ment_to_ent_scores))
		elif embed_type == "none": # useful for indexing methods that do not require label embeddings
			label_embeds = None
		else:
			raise Exception(f"embed_type = {embed_type} not supported")
		
		
		if graph_type in ["nsw", "hnsw"]:
			LOGGER.info(f"Building HNSW graph using entity reps : {label_embeds.shape}")
			dim = label_embeds.shape[1]
			index = HNSWWrapper(dim=dim, data=label_embeds, max_nbrs=max_nbrs, metric=graph_metric)
		elif graph_type == "knn":
			index = KNNGraph(data=label_embeds, max_nbrs=max_nbrs, metric=graph_metric)
		elif graph_type == "knn_e2e":
			index = KNNGraphwEnt2Ent(e2e_score_filename=e2e_score_filename, max_nbrs=max_nbrs)
		elif graph_type == "nsw_e2e":
			index = NSWGraphwEnt2Ent(
				e2e_score_filename=e2e_score_filename,
				max_nbrs=max_nbrs,
				n_long_range=int(np.ceil(n_long_range_frac*max_nbrs))
			)
		elif graph_type == "rand":
			index = RandomGraph(n_elems=len(label_embeds), max_nbrs=max_nbrs)
		else:
			raise NotImplementedError(f"Graph type = {graph_type} not supported")
		
		if index_path is not None:
			LOGGER.info(f"Now we will save the index at {index_path}")
			Path(os.path.dirname(index_path)).mkdir(exist_ok=True, parents=True)
			index.serialize(index_path)
			LOGGER.info("Finished serializing the object")

	return index




def get_init_ents(init_ent_method, k, n_ments, n_ents, ment_embeds, ent_embeds, force_exact_search):
	"""
	Find top-k entities for each mention using given embeddings and init_ent_method
	:param init_ent_method: Method for chosing initial entities
	:param k: Number of initial entities to choose
	:param n_ments: Number of mentions/inputs
	:param n_ents: Number of entities/labels
	:param ment_embeds: Embedding of all mentions
	:param ent_embeds: Embedding of all entities
	:return: List of list of initial entities for each mention. Shape: n_ments x k
	"""
	LOGGER.info(f"Creating initial entity entry points in the graph for each mention using method = {init_ent_method}")
	if init_ent_method == "random" or init_ent_method == "anchor":
		rng = np.random.default_rng(seed=0)
		init_ents = [rng.choice(n_ents, size=k, replace=False) for _ in range(n_ments)]
		return init_ents
	elif init_ent_method == "bienc" or init_ent_method == "tfidf":
		d = ment_embeds.shape[-1]
		LOGGER.info(f"Finding nearest entities for all mentions using embed of dim = {d}")
		# init_index = index = faiss.IndexFlatIP(d)
		# init_index.add(ent_embeds)
		# index = build_flat_or_ivff_index(embeds=ent_embeds, force_exact_search=False)
		index = build_flat_or_ivff_index(embeds=ent_embeds, force_exact_search=force_exact_search)
		_, init_ents = index.search(ment_embeds, k=k)
		return init_ents
	# elif init_ent_method == "hnsw":
	# 	# raise NotImplementedError(f"Init_ent_method = hnsw not implemented")
	# 	search_hnsw_graph(hnsw_graph, entity_scores, topk, beamsize, comp_budget)
	else:
		raise NotImplementedError(f"Init_ent_method = {init_ent_method} is not supported in get_init_ents function")
	


def search_nsw_graph(nsw_graph, entity_scores, topk, beamsize, init_ents, comp_budget, exit_at_local_minima_arg, pad_results):
	"""
	Seach over NSW graph to find top-scoring entities wrt given entity_scores
	:param nsw_graph: NSW graph to search over
	:param entity_scores: Score for each entity node in the graph or a function to compute these scores
	:param topk: Number of top scoring entities to find in the graph
	:param beamsize: Size of beam for greedy search over NSW graph
	:param init_ents: List of initial entry points in the graph
	:param comp_budget: Budget of node scores that we can compute during NSW search
	:param exit_at_local_minima_arg: Whether to terminate search when hitting a local minima even if we have
									computational budget. This is used only when comp_budget is an int/float.
									If comp_budget is None, then search WILL TERMINATE after hitting a local minima.
	:param pad_results: Pad top_k_ents array to contain topk number of entries even if less than topk entries are found
	:return:
	"""
	try:
		# Exit to local minima if comp_budget is None or exit_at_local_minima_arg is True
		# If comp_budget is None, then it means that we have explicit limit on number of score computations and thus
		# we exit at a local-minima to avoid exploring all nodes in the graph
		if comp_budget is None:
			LOGGER.info(f"Overriding exit_at_local_minima_arg = {exit_at_local_minima_arg} value and setting it to True")
		exit_at_local_minima =  (comp_budget is None) or exit_at_local_minima_arg
		
		if torch.is_tensor(entity_scores):
			entity_scores = entity_scores.cpu().numpy()
			
		
		n_ents = len(nsw_graph)
		num_score_comps = 0
		
		if init_ents is not None:
			curr_beam = init_ents[:beamsize]
			if len(curr_beam) != beamsize:
				warnings.warn(f"Initial beam is smaller than beam size as len(init_ents) = {len(init_ents)}")
				LOGGER.info(f"Initial beam is smaller than beam size as len(init_ents) = {len(init_ents)}")
		else:
			seed = 0
			rng = np.random.default_rng(seed=seed)
			curr_beam = rng.choice(n_ents, size=beamsize, replace=False)
		
		
		# First score all nodes in beam and add them to set of explored nodes
		if isinstance(entity_scores, np.ndarray):
			explored_nodes = {curr_node: entity_scores[curr_node] for curr_node in curr_beam}
		elif isinstance(entity_scores, types.FunctionType):
			beam_scores = entity_scores(curr_beam)
			explored_nodes = {curr_node: curr_score for curr_node, curr_score in zip(curr_beam, beam_scores)}
		else:
			raise NotImplementedError(f"entity_scores of type = {type(entity_scores)} not supported")
		
		num_score_comps += len(explored_nodes)
		
		topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=explored_nodes.items(), topk=topk)
		
		
		for search_iter in range(n_ents): # This is maximum number of search iterations we need
			
			# Break if we have exceeded our computation budget
			if (comp_budget is not None) and num_score_comps > comp_budget:
				break
				
			# Find best-nodes amongst nbrs of nodes in current beam
			curr_topk_nodes_n_scores, curr_num_score_comps = _search_nsw_graph_helper(nsw_graph=nsw_graph,
																					  entity_scores=entity_scores,
																					  topk=topk,
																					  curr_beam=curr_beam,
																					  explored_nodes=explored_nodes)
			
			num_score_comps += curr_num_score_comps
			
			
			# If all nbrs of nodes in beam are explored i.e, curr_topk_nodes_n_scores is empty
			#  - OR -
			# If we have found topk nodes AND
			#	If node with best score in current round is worse than worst node amongst top-k i.e. we are at local minima, AND
			# 		(
			#   		If exit_at_local_minima == True
			# 						- OR -
			# 			If we have exceeded our computation budget
			# 		)
			# then stop
			
			if len(curr_topk_nodes_n_scores) == 0:
				# If all nbrs of nodes in beam are explored i.e, curr_topk_nodes_n_scores is empty
				# This means that we have hit a dead end and not necessarily a local minima
				break
				
			if len(topk_nodes_n_scores) >= topk and curr_topk_nodes_n_scores[0][1] < topk_nodes_n_scores[-1][1] :
				if exit_at_local_minima \
						or (num_score_comps > comp_budget):
					break
			
			# Merge curr_topk_nodes with overall topk nodes and find topk
			topk_nodes_n_scores += curr_topk_nodes_n_scores
			topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=topk_nodes_n_scores, topk=topk)
			
			# Update beam best nodes found in this round
			curr_topk_nodes, _ = zip(*curr_topk_nodes_n_scores)
			curr_beam = list(curr_topk_nodes)[:beamsize]
			
		
		if len(topk_nodes_n_scores) < topk and pad_results: # If we are not able to find topk labels, then just repeat last label to make sure we return a top-k dim arrays
			topk_nodes_n_scores += [topk_nodes_n_scores[-1]]*topk
			topk_nodes_n_scores = topk_nodes_n_scores[:topk]
			
		assert len(topk_nodes_n_scores) == topk or n_ents < topk or (not pad_results), \
			f"Either number of entities ({n_ents}) should be less than topk ({topk})"
		
		topk_nodes, topk_scores  = zip(*topk_nodes_n_scores)
		return np.array(topk_scores), np.array(topk_nodes), num_score_comps
	except Exception as e:
		embed()
		raise e


def _search_nsw_graph_helper(nsw_graph, entity_scores, topk, curr_beam, explored_nodes):
	"""
	Searches over nbrs of nodes in beam and return topk neighbors that are not part of explored_nodes
	Modifies explored_nodes in place.
	:param nsw_graph: NSW graph to search over
	:param entity_scores: Score for each entity node in the graph or a function to compute these scores
	:param topk: Number of top scoring entities to find in the graph
	:param curr_beam: List of entities in current beam during search over NSW
	:param explored_nodes: List of entities that have been explored during search
	:return:
	"""
	try:
		num_score_comps = 0
		curr_node_n_score_tuples = []
		nodes_to_visit  = []
		for curr_node in curr_beam:
			# Iterate over all unexplored neighbors of this node and score them
			for curr_node_nbr in nsw_graph[curr_node]:
				if curr_node_nbr not in explored_nodes:
					nodes_to_visit += [curr_node_nbr]
					explored_nodes[curr_node_nbr] = -1
	
		# Score curr_node_nbrs wrt given mention
		if isinstance(entity_scores, np.ndarray):
			nodes_to_visit_scores = entity_scores[nodes_to_visit]
		elif isinstance(entity_scores, types.FunctionType):
			nodes_to_visit_scores = entity_scores(nodes_to_visit)
		else:
			raise Exception(f"Entity scores of type = {type(entity_scores)} not supported")
		
		assert len(nodes_to_visit_scores) == len(nodes_to_visit), f"Len of nodes_to_visit_scores ={len(nodes_to_visit_scores)} != len of nodes_to_visit = {len(nodes_to_visit)}"
		
		num_score_comps += len(nodes_to_visit)
		curr_node_n_score_tuples += list(zip(nodes_to_visit, nodes_to_visit_scores))
		explored_nodes.update({node:score for node, score in curr_node_n_score_tuples})
		
		# Pick top-k nodes to put in beam and search again
		topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=curr_node_n_score_tuples, topk=topk)
		return topk_nodes_n_scores, num_score_comps

	except Exception as e:
		embed()
		raise e


def search_nsw_graph_w_backtrack(nsw_graph, entity_scores, topk, beamsize, init_ents, comp_budget, exit_at_local_minima_arg, pad_results):
	"""
	Search over NSW graph to find top-scoring entities wrt given entity_scores.
	When finding next set of nodes to put in beam, it considers all nbrs nodes of previous beam which have not been
	made part of beam yet instead of using just top-k scored nbrs of previous beam.
	:param nsw_graph: NSW graph to search over
	:param entity_scores: Score for each entity node in the graph or a function to compute these scores
	:param topk: Number of top scoring entities to find in the graph
	:param beamsize: Size of beam for greedy search over NSW graph
	:param init_ents: List of initial entry points in the graph
	:param comp_budget: Budget of node scores that we can compute during NSW search
	:param exit_at_local_minima_arg: Whether to terminate search when hitting a local minima even if we have
									computational budget. This is used only when comp_budget is an int/float.
									If comp_budget is None, then search WILL TERMINATE after hitting a local minima.
	:param pad_results: Pad top_k_ents array to contain topk number of entries even if less than topk entries are found
	:return:
	"""
	try:
		# Exit to local minima if comp_budget is None or exit_at_local_minima_arg is True
		# If comp_budget is None, then it means that we have explicit limit on number of score computations and thus
		# we exit at a local-minima to avoid exploring all nodes in the graph
		if comp_budget is None:
			LOGGER.info("Overriding exit_at_local_minima_arg value and setting it to True")
		exit_at_local_minima =  (comp_budget is None) or exit_at_local_minima_arg
		
		if torch.is_tensor(entity_scores):
			entity_scores = entity_scores.cpu().numpy()
			
		
		n_ents = len(nsw_graph)
		num_score_comps = 0
		
		if init_ents is not None:
			curr_beam = init_ents[:beamsize]
			if len(curr_beam) != beamsize:
				warnings.warn(f"Initial beam is smaller than beam size as len(init_ents) = {len(init_ents)}")
				LOGGER.info(f"Initial beam is smaller than beam size as len(init_ents) = {len(init_ents)}")
		else:
			seed = 0
			rng = np.random.default_rng(seed=seed)
			curr_beam = rng.choice(n_ents, size=beamsize, replace=False)
		
		
		# First score all nodes in beam and add them to set of explored nodes
		if isinstance(entity_scores, np.ndarray):
			scored_nodes = {curr_node: entity_scores[curr_node] for curr_node in curr_beam}
		elif isinstance(entity_scores, types.FunctionType):
			beam_scores = entity_scores(curr_beam)
			scored_nodes = {curr_node: curr_score for curr_node, curr_score in zip(curr_beam, beam_scores)}
		else:
			raise NotImplementedError(f"entity_scores of type = {type(entity_scores)} not supported")
		
		num_score_comps += len(scored_nodes)
		
		topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=scored_nodes.items(), topk=topk)
		nbr_explored_nodes = {} # Nodes whose nbrs have been explored during search
		
		for search_iter in range(n_ents): # This is maximum number of search iterations we need
			
			# Break if we have exceeded our computation budget
			if (comp_budget is not None) and num_score_comps > comp_budget:
				break
				
			# Find best-nodes amongst nbrs of nodes in current beam
			(
				curr_topk_nodes_n_scores,
				curr_unvisited_nbr_nodes,
				curr_num_score_comps
			) = _search_nsw_graph_helper(
				nsw_graph=nsw_graph,
				entity_scores=entity_scores,
				topk=topk,
				curr_beam=curr_beam,
				scored_nodes=scored_nodes,
				nbr_explored_nodes=nbr_explored_nodes
			)
			
			num_score_comps += curr_num_score_comps
			
			
			# If all nbrs of nodes in beam are explored i.e, curr_topk_nodes_n_scores is empty
			#  - OR -
			# If we have found topk nodes AND
			#	If node with best score in current round is worse than worst node amongst top-k i.e. we are at local minima, AND
			# 		(
			#   		If exit_at_local_minima == True
			# 						- OR -
			# 			If we have exceeded our computation budget
			# 		)
			# then stop
			
			# if len(curr_topk_nodes_n_scores) == 0:
			# 	# If all nbrs of nodes in beam are explored i.e, curr_topk_nodes_n_scores is empty
			# 	# This means that we have hit a dead end and not necessarily a local minima
			# 	break
			
			if len(curr_unvisited_nbr_nodes) == 0:
				# If all nbrs of nodes in beam have been part of beam visited i.e, curr_topk_nodes_n_scores is empty
				# This means that we have hit a dead end in that all nbrs of current beam have been explored
				# so there are no new nodes to hop on to for search.
				break
				
			if len(topk_nodes_n_scores) >= topk and curr_topk_nodes_n_scores[0][1] < topk_nodes_n_scores[-1][1]:
				if exit_at_local_minima \
						or (num_score_comps > comp_budget):
					break
			
			# Merge curr_topk_nodes with overall topk nodes and find topk
			topk_nodes_n_scores += curr_topk_nodes_n_scores
			topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=topk_nodes_n_scores, topk=topk)
			
			# # Update beam best nodes found in this round
			# curr_topk_nodes, _ = zip(*curr_topk_nodes_n_scores)
			# curr_beam = list(curr_topk_nodes)[:beamsize]
		
			# Choose next beam from nodes in nbrhood of current beam and
			# nodes that have not been visited yet (although they might have been scored)
			curr_unvisited_nbr_nodes_n_scores = _get_topk(
				curr_node_n_score_tuples=[(n, scored_nodes[n]) for n in curr_unvisited_nbr_nodes],
				topk=len(curr_unvisited_nbr_nodes)
			)
			curr_beam, _ = zip(*curr_unvisited_nbr_nodes_n_scores)
			curr_beam = list(curr_beam)[:beamsize]
			
		
		if len(topk_nodes_n_scores) < topk and pad_results: # If we are not able to find topk labels, then just repeat last label to make sure we return a top-k dim arrays
			topk_nodes_n_scores += [topk_nodes_n_scores[-1]]*topk
			topk_nodes_n_scores = topk_nodes_n_scores[:topk]
			
		assert len(topk_nodes_n_scores) == topk or n_ents < topk or (not pad_results), \
			f"Either number of entities ({n_ents}) should be less than topk ({topk}) " \
			f"or pad_results = {pad_results} should be set to false"
		
		topk_nodes, topk_scores  = zip(*topk_nodes_n_scores)
		return np.array(topk_scores), np.array(topk_nodes), num_score_comps
	except Exception as e:
		embed()
		raise e


def _search_nsw_graph_w_backtrack_helper(nsw_graph, entity_scores, topk, curr_beam, scored_nodes, nbr_explored_nodes):
	"""
	Searches over nbrs of nodes in beam and return topk neighbors that are not part of explored_nodes
	Modifies scored_nodes and nbr_explord_nodes in place.
	When finding next set of nodes to put in beam, it considers all nbrs nodes of previous beam which have not been
	made part of beam yet instead of using just top-k scored nbrs of previous beam.
	
	:param nsw_graph: NSW graph to search over
	:param entity_scores: Score for each entity node in the graph or a function to compute these scores
	:param topk: Number of top scoring entities to find in the graph
	:param curr_beam: List of entities in current beam during search over NSW
	:param scored_nodes: List of entities that have been scored during search
	:param nbr_explored_nodes: List of nodes whose nbrs have been scored during search i.e. these nodes have already been part of the beam at some point
	:return: Top-k nodes amongst nbrs of current node
	"""
	try:
		num_score_comps = 0
		nodes_to_score  = [] # List of nodes that haven't been scored yet
		nodes_to_visit	= {} # Set of nodes whose nbrs haven't been explored yet. These nodes themselves might have been scored already.
		all_nbrs = {}
		for curr_node in curr_beam:
			nbr_explored_nodes[curr_node]  = 1
			# Iterate over all neighbors of this node and find list of nbrs to score
			for curr_node_nbr in nsw_graph[curr_node]:
				all_nbrs[curr_node_nbr] = 1
				if curr_node_nbr not in scored_nodes:
					nodes_to_score += [curr_node_nbr]
					scored_nodes[curr_node_nbr] = -1
				
				# If this node's nbrs have not been explored, then add this to nodes_to_visit.
				# This is used for picking beam in next iteration of searcg
				if curr_node_nbr not in nbr_explored_nodes:
					nodes_to_visit[curr_node_nbr] = 1
	
		# Score nbrs of nodes in beam that haven't been scored wrt given mention
		if isinstance(entity_scores, np.ndarray):
			curr_beam_nbr_scores = entity_scores[nodes_to_score]
		elif isinstance(entity_scores, types.FunctionType):
			curr_beam_nbr_scores = entity_scores(nodes_to_score)
		else:
			raise Exception(f"Entity scores of type = {type(entity_scores)} not supported")
		
		assert len(curr_beam_nbr_scores) == len(nodes_to_score), f"Len of nodes_to_visit_scores = {len(curr_beam_nbr_scores)} != len of nodes_to_visit = {len(nodes_to_score)}"
		
		num_score_comps += len(nodes_to_score)
		
		curr_beam_nbr_node_n_score_tuples = list(zip(nodes_to_visit, curr_beam_nbr_scores))
		
		# Update scored_nodes dict with actual scores for each node scored in this round
		scored_nodes.update({node:score for node, score in curr_beam_nbr_node_n_score_tuples})
		
		# # Pick top-k explored nbr nodes explored in this round.
		# # Including other nbrs nodes which have been explored in previous round will not affect eventual performance
		# # as if one of those nodes is part of global top-k, then it had been already added to global list of topk
		# # and if it is not part of global top-k, adding this to current top-k nodes will not change global topk anyway.
		# topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=curr_beam_nbr_node_n_score_tuples, topk=topk)
		
		# Find top-k nbrs
		nbrs_nodes_n_scores = [(n, scored_nodes[n]) for n in all_nbrs]
		topk_nodes_n_scores = _get_topk(curr_node_n_score_tuples=nbrs_nodes_n_scores, topk=topk)
		
		return topk_nodes_n_scores, list(nodes_to_visit.keys()), num_score_comps

	except Exception as e:
		embed()
		raise e


def _get_comp_budget(num_nodes, level, base_comp_budget):
	
	"""
	Budget -
	If number of nodes <= 20: exhaustive search
	If number of nodes > 20: Search max(10, 10%num_nodes)
	"""
	if level == 1:
		return base_comp_budget
	else:
		if num_nodes <= 20:
			return num_nodes
		else:
			return max(10, int(0.1*num_nodes))
	
	
def search_hnsw_graph(hnsw_graph, entity_scores, topk, beamsize, comp_budget, exit_at_local_minima_arg, pad_results):
	"""
	Run search over HNSW graph
	:param hnsw_graph: List of NSW graphs with first element corresponding to top-level graph
	:param entity_scores:
	:param topk:
	:param beamsize:
	:param comp_budget:
	:param exit_at_local_minima_arg:
	:param pad_results
	:return:
	"""
	LOGGER.info(f"Beginning HNSW search with params "
				f"topk = {topk} "
				f"beamsize = {beamsize} "
				f"comp_budget = {comp_budget} "
				f"exit_at_local_minima_arg = {exit_at_local_minima_arg} "
				f"pad_results = {pad_results}")
	
	higher_level_beamsize = min(2, beamsize)
	n_levels = len(hnsw_graph)
	
	nsw_topk_scores, nsw_topk_ents, nsw_curr_num_score_comps = [], [], 0
	
	# TODO: Refine logic for deciding on init_ents
	curr_init_ents = [n for n, nbrs in hnsw_graph[0].items() if len(nbrs) > 0]
	curr_init_ent_scores = [entity_scores[n] for n in curr_init_ents]
	curr_ents_n_scores = sorted(zip(curr_init_ents, curr_init_ent_scores), key=lambda x:x[1], reverse=True)
	curr_init_ents, curr_init_ent_scores = zip(*curr_ents_n_scores)
	curr_init_ents = list(curr_init_ents)
	
	for curr_depth, curr_nsw_graph in enumerate(hnsw_graph):
		
		curr_level = n_levels - curr_depth
		curr_num_nodes = len([1 for nbrs in curr_nsw_graph.values() if len(nbrs) > 0])
		
		assert curr_level > 0 and curr_level <= n_levels, f"curr_level value = {curr_level} does not fit in [1,{n_levels}]"
		
		if curr_level == 1: # Lowest level containing NSW over all nodes
			curr_topk = topk
			curr_beamsize = beamsize
			curr_comp_budget = comp_budget
			curr_exit_at_local_minima_arg = exit_at_local_minima_arg
			curr_pad_results = pad_results
		else:
			curr_topk = min(curr_num_nodes, beamsize)
			curr_beamsize = higher_level_beamsize
			# curr_comp_budget = curr_num_nodes if curr_num_nodes <= 20 else max(10, int(0.1*curr_num_nodes))
			# curr_exit_at_local_minima_arg = True
			curr_comp_budget = comp_budget
			curr_exit_at_local_minima_arg = False
			curr_pad_results = False
		
		LOGGER.info(f"Searching level = {curr_level}, depth = {curr_depth}, of total level = {n_levels}"
					f"num_nodes = {curr_num_nodes}, curr_beamsize = {curr_beamsize} and curr_topk = {curr_topk}")
		
		# Search over curr_nsw_graph
		(nsw_topk_scores,
		 nsw_topk_ents,
		 nsw_curr_num_score_comps) = search_nsw_graph(
			entity_scores=entity_scores,
			nsw_graph=curr_nsw_graph,
			topk=curr_topk,
			beamsize=curr_beamsize,
			init_ents=curr_init_ents,
			comp_budget=curr_comp_budget,
			exit_at_local_minima_arg=curr_exit_at_local_minima_arg,
			pad_results=curr_pad_results
		)
		
		# Use search results to init search for graph at lower level
		curr_init_ents = nsw_topk_ents
	
	return nsw_topk_scores, nsw_topk_ents, nsw_curr_num_score_comps

	


def run_hnsw_search(
		gt_labels,
		hnsw_graph,
		ment_to_ent_scores,
		rerank_ment_to_ent_scores,
		topk,
		beamsize,
		init_ents,
		comp_budget,
		exit_at_local_minima_arg,
		run_only_nsw_search,
		graph_type,
		ment_idxs_to_use
):
	"""
	Run NSW search and exact search, and finally evaluate accuracy of both methods
	:param init_ents: List of initial entities for each mention. Shape n_ments x num_init_ents
	:param beamsize: Size of beam for greedy search over NSW
	:param topk: Number of top scoring entities to find in the graph
	:param rerank_ment_to_ent_scores:
	:param hnsw_graph: List of NSW graphes where each graph on entities stored in adjacency list format
	:param ment_to_ent_scores: n_ments x n_ents shape matrix storing scores for each mention,entity pair
	:return: Dict containing eval results for NSW and exact search
	"""
	
	n_ments, n_ents = ment_to_ent_scores.shape
	
	assert all(len(nsw_graph) == n_ents for nsw_graph in hnsw_graph), \
		f"Number of entities in NSW graph = {[len(nsw_graph) for nsw_graph in hnsw_graph]} " \
		f"does not match that in score matrix = {n_ents}"
	
	exact_topk_preds = []
	exact_topk_reranked_preds = []
	nsw_topk_preds = []
	nsw_topk_reranked_preds = []
	init_topk_preds = []
	
	nsw_num_score_comps = []
	for ment_idx in range(n_ments):
		if ment_idx not in ment_idxs_to_use: continue # FIXME: Remove this
		
		entity_scores = ment_to_ent_scores[ment_idx]
		# Get top-k indices from using NSW index
		if run_only_nsw_search: # TODO: Move if this condition in search_hnsw_graph function
			(nsw_topk_scores,
			 nsw_topk_ents,
			 nsw_curr_num_score_comps) = search_nsw_graph(
				entity_scores=entity_scores,
				nsw_graph=hnsw_graph[-1],
				topk=topk,
				beamsize=beamsize,
				init_ents=init_ents[ment_idx],
				comp_budget=comp_budget,
				exit_at_local_minima_arg=exit_at_local_minima_arg,
				pad_results=True
			)
		else:
			(nsw_topk_scores,
			 nsw_topk_ents,
			 nsw_curr_num_score_comps) = search_hnsw_graph(
				entity_scores=entity_scores,
				hnsw_graph=hnsw_graph,
				topk=topk,
				beamsize=beamsize,
				comp_budget=comp_budget,
				exit_at_local_minima_arg=exit_at_local_minima_arg,
				pad_results=True
			)
		
		init_topk_ents = init_ents[ment_idx][:topk]
		init_topk_scores = np.array(list(range(topk, 0, -1)))
		
		# Get top-k indices from exact matrix
		exact_topk_scores, exact_topk_ents = entity_scores.topk(topk)
		
		# Re-rank top-k indices from NSW
		temp = torch.zeros(entity_scores.shape) - 99999999999999
		temp[nsw_topk_ents] = rerank_ment_to_ent_scores[ment_idx][nsw_topk_ents]
		nsw_topk_reranked_scores, nsw_topk_reranked_ents = temp.topk(topk)

		
		# Re-rank top-k indices from exact matrix
		temp = torch.zeros(entity_scores.shape) - 99999999999999
		temp[exact_topk_ents] = rerank_ment_to_ent_scores[ment_idx][exact_topk_ents]
		exact_topk_reranked_scores, exact_topk_reranked_ents = temp.topk(topk)
		
		init_topk_preds += [(init_topk_ents, init_topk_scores)]
	
		nsw_topk_preds += [(nsw_topk_ents, nsw_topk_scores)]
		exact_topk_preds += [(exact_topk_ents, exact_topk_scores)]
		
		nsw_topk_reranked_preds += [(nsw_topk_reranked_ents, nsw_topk_reranked_scores)]
		exact_topk_reranked_preds += [(exact_topk_reranked_ents, exact_topk_reranked_scores)]
		
		nsw_num_score_comps += [nsw_curr_num_score_comps]
		
		
	init_topk_preds = _get_indices_scores(init_topk_preds)
	
	nsw_topk_preds = _get_indices_scores(nsw_topk_preds)
	exact_topk_preds = _get_indices_scores(exact_topk_preds)
	
	nsw_topk_reranked_preds = _get_indices_scores(nsw_topk_reranked_preds)
	exact_topk_reranked_preds = _get_indices_scores(exact_topk_reranked_preds)
	
	
	res = {"init":score_topk_preds(gt_labels=gt_labels,
								   topk_preds={"indices":init_topk_preds["indices"],
												  "scores":init_topk_preds["scores"]}),
		   graph_type: score_topk_preds(gt_labels=gt_labels,
								   		topk_preds={"indices":nsw_topk_preds["indices"],
												    "scores":nsw_topk_preds["scores"]}),
		   "exact": score_topk_preds(gt_labels=gt_labels,
									 topk_preds={"indices":exact_topk_preds["indices"],
												 "scores":exact_topk_preds["scores"]}),
		   f"{graph_type}_reranked": score_topk_preds(gt_labels=gt_labels,
												 	  topk_preds={"indices":nsw_topk_reranked_preds["indices"],
															 	  "scores":nsw_topk_reranked_preds["scores"]}),
		   "exact_reranked": score_topk_preds(gt_labels=gt_labels,
											  topk_preds={"indices":exact_topk_reranked_preds["indices"],
														  "scores":exact_topk_reranked_preds["scores"]}),
		   f"exact_vs_{graph_type}": compute_overlap(indices_list1=exact_topk_preds["indices"],
													 indices_list2=nsw_topk_preds["indices"])
		   }
	new_res = {f"{res_type}~{metric}":res[res_type][metric]
			   for res_type in res
				for metric in res[res_type]}
	
	new_res[f"{graph_type}~num_score_comps~mean"] = np.mean(nsw_num_score_comps)
	new_res[f"{graph_type}~num_score_comps~std"] = np.std(nsw_num_score_comps)
	for _centile in [1, 10, 50, 90, 99]:
		new_res[f"{graph_type}~num_score_comps~p{_centile}"] = np.percentile(nsw_num_score_comps, _centile)
	
	return new_res
	

def run(embed_type, res_dir, data_info, bi_model_file, graph_metric, graph_type, entry_method, force_exact_init_search,
		e2e_score_filename, a2e_score_filename, max_nbr_vals, topk_vals, beamsize_vals, budget_vals, misc, arg_dict):
	try:
		wandb.run.summary["status"] = 1
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fname = data_info
		
		wandb.run.summary["status"] = 2
		############################### Read pre-computed cross-encoder score matrix ###################################
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
		
			# FIXME: REMOVE THIS IF CONDITION
			if "rank_margin_neg" in data_fname["crossenc_ment_to_ent_scores"]:
				crossenc_ment_to_ent_scores = -1*crossenc_ment_to_ent_scores
		
		# FIXME: Remove this hack here - This is to only run exps for embed_type = anchor
		# anc_ment_to_ent_scores  = crossenc_ment_to_ent_scores[10:, :]
		# crossenc_ment_to_ent_scores  = crossenc_ment_to_ent_scores[:10, :]
		# test_data = test_data[:10]
		# mention_tokens_list = mention_tokens_list[:10]
		
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
		
		ment_idxs_to_use = set(list(range(n_ments)))
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]]) for ment_dict in test_data]
		
		wandb.run.summary["status"] = 3
		############################## Compute mention and entity embeddings #################################################
		
		# Load biencoder model if required
		if bi_model_file.endswith(".json"):
			with open(bi_model_file, "r") as fin:
				biencoder = BiEncoderWrapper.load_model(config=json.load(fin))
		else: # Load from pytorch lightning checkpoint
			try:
				biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
			except:
				LOGGER.info("Loading biencoder trained with shared params with a cross-encoder")
				biencoder = CrossEncoderWrapper.load_from_checkpoint(bi_model_file)
				
		biencoder.eval()

		# 1.)  Compute mention embeddings to use for choosing entry points in graph
		LOGGER.info(f"Computing mention encodings computed using method = {embed_type}")
		ment_embeds = compute_ment_embeds(
			embed_type=entry_method,
			entity_file=data_fname["ent_file"],
			mentions=mentions,
			biencoder=biencoder,
			mention_tokens_list=mention_tokens_list
		)
		
		# 2.)  Compute entity embeddings to use for choosing entry points in graph
		LOGGER.info(f"Computing entity encodings computed using method = {embed_type}")
		ent_embeds_for_init = compute_ent_embeds(
			embed_type=entry_method,
			biencoder=biencoder,
			entity_tokens_file=data_fname["ent_tokens_file"],
			entity_file=data_fname["ent_file"],
		)
		
		# 3.)  Compute entity embeddings to use for building graph index
		LOGGER.info(f"Computing entity encodings computed using method = {embed_type}")
		if embed_type == entry_method:
			ent_embeds_for_index = ent_embeds_for_init
		elif embed_type == "bienc" and entry_method != "bienc":
			ent_embeds_for_index = compute_ent_embeds(
				embed_type=embed_type,
				biencoder=biencoder,
				entity_tokens_file=data_fname["ent_tokens_file"],
				entity_file=data_fname["ent_file"],
			)
		elif embed_type == "c-anchor":
			LOGGER.info("Loading anchor to entity scores")
			with open(a2e_score_filename, "rb") as fin:
				dump_dict = pickle.load(fin)
				ent_embeds_for_index = dump_dict["ent_to_ent_scores"] # Shape : Number of entities x Number of anchors
				topk_ents = set(dump_dict["topk_ents"][0]) # Shape : Number of anchors
				
			if torch.is_tensor(ent_embeds_for_index):
				ent_embeds_for_index = ent_embeds_for_index.cpu().detach().numpy()
			assert ent_embeds_for_index.shape[0] == n_ents, f"ent_embeds_for_index.shape[0] = {ent_embeds_for_index.shape[0]} != n_ents = {n_ents}"
			
			new_gt_labels = []
			for ctr, _gt_label in enumerate(curr_gt_labels):
				if _gt_label not in topk_ents:
					ment_idxs_to_use.remove(ctr)
				else:
					new_gt_labels += [_gt_label]
			curr_gt_labels = new_gt_labels
		else:
			ent_embeds_for_index = None
		
		
		# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
		ent_embeds_for_init = ent_embeds_for_init[entity_id_list] if ent_embeds_for_init is not None else None
		ent_embeds_for_index = ent_embeds_for_index[entity_id_list] if ent_embeds_for_index is not None else None
		
		# if embed_type != "bienc":
		# 	LOGGER.info(f"Computing entity encodings computed using method = {embed_type}")
		# 	if bi_model_file.endswith(".json"):
		# 		with open(bi_model_file, "r") as fin:
		# 			biencoder = BiEncoderWrapper.load_model(config=json.load(fin))
		# 	else: # Load from pytorch lightning checkpoint
		# 		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
		# 	biencoder.eval()
		#
		# 	bienc_ent_embeds = compute_ent_embeds(
		# 		embed_type="bienc",
		# 		biencoder=biencoder,
		# 		entity_tokens_file=data_fname["ent_tokens_file"],
		# 		entity_file=data_fname["ent_file"],
		# 	)
		# 	bienc_ent_embeds = torch.Tensor(bienc_ent_embeds)
		# 	bienc_ent_embeds = bienc_ent_embeds[entity_id_list]
		# else:
		# 	bienc_ent_embeds =  torch.Tensor(ent_embeds)

		# bienc_ment_to_ent_scores = compute_ment_to_ent_matrix_w_bienc(biencoder, mention_tokens_list, bienc_ent_embeds)
		################################################################################################################
		
		result = {}
		
		exit_at_local_minima_arg = False
		
		wandb.run.summary["status"] = 4
		for max_nbr_ctr, max_nbrs in enumerate(max_nbr_vals):
			wandb.log({"max_nbr_ctr": max_nbr_ctr})
			######################################## Build/Read NSW Index on entities ######################################
			index_path = None # Passing None so that we do not save the index or load a pre-computed one
			index = get_index(
				index_path=index_path,
				embed_type=embed_type,
				entity_file=data_fname["ent_file"],
				bienc_ent_embeds=ent_embeds_for_index,
				ment_to_ent_scores=crossenc_ment_to_ent_scores,
				max_nbrs=max_nbrs,
				graph_metric=graph_metric,
				graph_type=graph_type,
				e2e_score_filename=e2e_score_filename
			)
			
			LOGGER.info(f"Built graph with max_nbrs = {max_nbrs}")
			levels = index.get_levels()
			n_levels = max(levels)
			# for curr_level in range(1, max(levels)+1):
			# 	LOGGER.info(f"{curr_level}\t{len([i for i,l in enumerate(levels) if l>=curr_level])}")
			# LOGGER.info("")
			
			
			LOGGER.info(f"Extracting (H)NSW graph from index with max_nbrs={max_nbrs}")
			# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
			# nsw_graph = index.get_nsw_graph_at_level(level=1)
			hnsw_graph = [index.get_nsw_graph_at_level(level=curr_level) for curr_level in range(n_levels, 0, -1)]
			################################################################################################################
			
			LOGGER.info("Now we will search over the graph")
			
			for init_ent_method, comp_budget in itertools.product([entry_method], budget_vals):
				init_ents = get_init_ents(
					init_ent_method=init_ent_method,
					k=max(beamsize_vals),
					ment_embeds=ment_embeds,
					ent_embeds=ent_embeds_for_init,
					n_ments=n_ments,
					n_ents=n_ents,
					force_exact_search=force_exact_init_search
				)
				LOGGER.info("Now beginning search")
				for topk, beamsize in tqdm(itertools.product(topk_vals, beamsize_vals)):
					key = f"k={topk}_b={beamsize}_init_{init_ent_method}_budget={comp_budget}_max_nbrs={max_nbrs}"
					result[key] = {}

					crossenc_result = run_hnsw_search(
						hnsw_graph=hnsw_graph,
						topk=topk,
						beamsize=beamsize,
						init_ents=init_ents,
						comp_budget=comp_budget,
						gt_labels=curr_gt_labels,
						ment_to_ent_scores=crossenc_ment_to_ent_scores,
						rerank_ment_to_ent_scores=crossenc_ment_to_ent_scores,
						exit_at_local_minima_arg=exit_at_local_minima_arg,
						run_only_nsw_search=True,
						graph_type=graph_type,
						ment_idxs_to_use=ment_idxs_to_use
					)
					result[key].update({"crossenc~"+k:v for k,v in crossenc_result.items()})
					
					# crossenc_result = run_hnsw_search(
					# 	hnsw_graph=hnsw_graph,
					# 	topk=topk,
					# 	beamsize=beamsize,
					# 	init_ents=init_ents,
					# 	comp_budget=comp_budget,
					# 	gt_labels=curr_gt_labels,
					# 	ment_to_ent_scores=crossenc_ment_to_ent_scores,
					# 	rerank_ment_to_ent_scores=crossenc_ment_to_ent_scores,
					# 	exit_at_local_minima_arg=exit_at_local_minima_arg,
					# 	run_only_nsw_search=False
					# )
					# result[key].update({"crossenc~"+k:v for k,v in crossenc_result.items()})
					
					# bienc_result = run_hnsw_search(
					# 	hnsw_graph=hnsw_graph,
					# 	topk=topk,
					# 	beamsize=beamsize,
					# 	init_ents=init_ents,
					# 	comp_budget=comp_budget,
					# 	gt_labels=curr_gt_labels,
					# 	ment_to_ent_scores=bienc_ment_to_ent_scores,
					# 	rerank_ment_to_ent_scores=crossenc_ment_to_ent_scores,
					# 	exit_at_local_minima_arg=exit_at_local_minima_arg,
					# 	run_only_nsw_search=True,
					# 	graph_type=graph_type
					# )
					# result[key].update({"bienc~"+k:v for k,v in bienc_result.items()})
					
		wandb.run.summary["status"] = 5
		
		
		# res_file = f"{res_dir}/nsw_eval_{data_name}_emb={embed_type}{misc}.json"
		res_file = f"{res_dir}/search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}{misc}.json"
		with open(res_file, "w") as fout:
			result["data_info"] = data_info
			result["arg_dict"] = arg_dict
			result["other_args"] = {
				"max_nbr_vals":max_nbr_vals,
				"topk_vals":topk_vals,
				"beamsize_vals":beamsize_vals,
				"budget_vals":budget_vals
			}
			json.dump(result, fout, indent=4)
			LOGGER.info(json.dumps(result,indent=4))
	except Exception as e:
		embed()
		raise e


def plot(res_dir, data_info, misc, max_nbr_vals, topk_vals, beamsize_vals, budget_vals, graph_type, graph_metric):
	try:
		embed_types = ["tfidf", "bienc", "anchor"]
		embed_types = ["tfidf", "bienc", "anchor"]
		entry_methods = ["tfidf", "bienc", "random"]
		# embed_types = ["bienc"]
		# embed_types = ["tfidf"]
		data_name, data_fnames = data_info
		plt_dir = f"{res_dir}/search_plots"
		Path(plt_dir).mkdir(exist_ok=True, parents=True)
		
		all_results = {}
		for embed_type, entry_method in itertools.product(embed_types, entry_methods):
			# res_file = f"{res_dir}/nsw_eval_{data_name}_emb={embed_type}{misc}.json" # Old NSW eval result file format
			res_file = f"{res_dir}/search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}{misc}.json"
			try:
				with open(res_file, "r") as fin:
					all_results[(embed_type, entry_method)] = json.load(fin)
			except FileNotFoundError:
				LOGGER.info(f"Result file for embed_type={embed_type}, entry_methohd={entry_method}, \n{res_file} not found")
				
		LOGGER.info(f"Result details found for - {all_results.keys()}")
		
		
		# Compare recall of labels wrt exact crossencoder model by nsw search vs number of computations
		# embed_types = ["bienc"]
		color_grad_variable_vals = ["max_nbrs", "beamsize", ""]
		
		topk_vals = [1, 64, 100]
		for color_grad_variable, topk  in itertools.product(color_grad_variable_vals, topk_vals):
			_plot_nsw_crossenc_recall_vs_comp_budget(
				all_results=all_results,
				plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}",
				model_type="crossenc",
				topk=topk,
				color_grad_variable=color_grad_variable,
				graph_type=graph_type,
				max_nbr_vals=max_nbr_vals,
				beamsize_vals=beamsize_vals,
				budget_vals=budget_vals,
			)
			# _plot_nsw_crossenc_recall_vs_comp_budget(
			# 	all_results=all_results,
			# 	plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}",
			# 	model_type="bienc",
			# 	topk=topk,
			# 	color_grad_variable=color_grad_variable,
			# 	graph_type=graph_type,
			# 	max_nbr_vals=max_nbr_vals,
			# 	beamsize_vals=beamsize_vals,
			# 	budget_vals=budget_vals
			# )
		
		# topk_vals = [64]
		# for color_grad_variable, topk  in itertools.product(color_grad_variable_vals, topk_vals):
		# 	_plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}", model_type="crossenc", topk=topk, metric="acc", color_grad_variable=color_grad_variable)
		# 	# # _plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact", model_type="bienc", topk=topk, metric="acc", color_grad_variable=color_grad_variable)
		#
		# 	_plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}", model_type="crossenc", topk=topk, metric="recall", color_grad_variable=color_grad_variable)
		# 	# # _plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact", model_type="bienc", topk=topk, metric="recall", color_grad_variable=color_grad_variable)
		
		
	except Exception as e:
		embed()
		raise e
		


def _plot_nsw_crossenc_recall_vs_comp_budget(
		all_results,
		plt_dir,
		model_type,
		topk,
		color_grad_variable,
		graph_type,
		max_nbr_vals,
		beamsize_vals,
		budget_vals
):
	
	try:
		cmap_vals = itertools.cycle(["Reds", "Greens", "Blues", "Greys", "Oranges", "Purples"])
		colors = itertools.cycle([
			('firebrick', 'lightsalmon'),
			('green', 'yellowgreen'),
			('navy', 'skyblue'),
			('gray', 'silver'),
			('olive', 'y'),
			('darkorange', 'gold'),
			('deeppink', 'violet'),
			('sienna', 'tan'),
			('darkviolet', 'orchid'),
			('deepskyblue', 'lightskyblue'),
		])


		embed_types_n_entry_methods = list(all_results.keys())
		# budget_vals = [b for b in budget_vals if b is not None]
		
		plt.clf()
		figure, axis = plt.subplots(1,1, figsize=(9,6))
		
		for (embed_type, entry_method), c, cmap in zip(embed_types_n_entry_methods, colors,  cmap_vals):
			X, Y, color_vals = [], [], []
			retr_X, retr_Y = [], []
		
			for beamsize, max_nbrs in itertools.product(beamsize_vals, max_nbr_vals):
				retr_Y += [ float(all_results[(embed_type, entry_method)][f"k={topk}_b={beamsize}_init_{entry_method}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0][5:]) ]
				retr_X += [ all_results[(embed_type, entry_method)][f"k={topk}_b={beamsize}_init_{entry_method}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"] ]

				curr_X = [ all_results[(embed_type, entry_method)][f"k={topk}_b={beamsize}_init_{entry_method}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"]
					  for budget in budget_vals]

				curr_Y = [ all_results[(embed_type, entry_method)][f"k={topk}_b={beamsize}_init_{entry_method}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0]
					  for budget in budget_vals]
				curr_Y = [float(y[5:]) for y in curr_Y]

				X += curr_X
				Y += curr_Y
				if color_grad_variable == "max_nbrs":
					color_vals += [np.log(max_nbrs) if max_nbrs is not None else np.log(64)]*len(budget_vals)
				elif color_grad_variable == "beamsize":
					color_vals += [np.log(beamsize+1)] *len(budget_vals)
				else:
					color_vals += [c[1]]*len(budget_vals)
				
			edgecolors = [c[1]]*len(X)
			axis.scatter(X, Y, c=color_vals, edgecolors=edgecolors, alpha=0.7, cmap=plt.get_cmap(cmap))
			axis.scatter(retr_X, retr_Y, label=f"{graph_type}~{embed_type}~{entry_method}", marker="x", c=c[0], alpha=0.8)
			temp ={x:y for x,y in zip(retr_X,retr_Y)}
			LOGGER.info(f"topk={topk}, embed_type: {embed_type} {temp} ")
		
		
		axis.set_ylim(0,1)
		axis.set_xlabel(f"Number of {model_type} calls")
		axis.set_ylabel(f"Recall wrt exact {model_type} model")
		domain = all_results[embed_types_n_entry_methods[0]]["data_info"][0]
		try:
			n_ent = int(all_results[embed_types_n_entry_methods[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1][:-4])
		except ValueError:
			n_ent = int(all_results[embed_types_n_entry_methods[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1].split("_all_layers")[0])
			
		figure.suptitle(f"Recall wrt exact {model_type} for domain {domain} w/ {n_ent} entities")
		axis.legend()
		axis.grid()
		out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlinear{color_grad_variable}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		# plt.savefig(out_file)
		
		axis.set_xscale("log")
		out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlog{color_grad_variable}.pdf"
		plt.savefig(out_file)
		
		# [ax.set_xlim(0.1,1000) for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlog_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		#
		# [ax.set_xscale("linear") for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlinear_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		plt.close()
	except Exception as e:
		embed()
		raise e



def plot_old(res_dir, data_info, misc, max_nbr_vals, topk_vals, beamsize_vals, budget_vals, graph_type, graph_metric):
	try:
		embed_types = ["tfidf", "bienc", "anchor"]
		embed_types = ["tfidf", "bienc"]
		entry_methods = ["tfidf", "bienc", "random"]
		# embed_types = ["bienc"]
		# embed_types = ["tfidf"]
		data_name, data_fnames = data_info
		plt_dir = f"{res_dir}/search_plots"
		Path(plt_dir).mkdir(exist_ok=True, parents=True)
		
		all_results = {}
		for embed_type, entry_method in itertools.product(embed_types, entry_methods):
			# res_file = f"{res_dir}/nsw_eval_{data_name}_emb={embed_type}{misc}.json" # Old NSW eval result file format
			res_file = f"{res_dir}/search_eval_{data_name}_g={graph_type}_{graph_metric}_e={embed_type}_init={entry_method}{misc}.json"
			try:
				with open(res_file, "r") as fin:
					all_results[embed_type] = json.load(fin)
			except FileNotFoundError:
				LOGGER.info(f"Result file for embed_type={embed_type}, entry_methohd={entry_method}, \n{res_file} not found")
				
		LOGGER.info(f"Result details found for - {all_results.keys()}")
		
		
		# # Compare different ways of building a graph
		# _plot_graph_const_comparison(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/graph_construction")
		
		# # Compare different ways of searching the graph
		# _plot_search_comparison(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/search_w_diff_methods")
		
		# Compare recall of labels wrt exact crossencoder model by nsw search vs number of computations
		# embed_types = ["bienc"]
		color_grad_variable_vals = ["max_nbrs", "beamsize", ""]
		
		topk_vals = [1, 64, 100]
		for color_grad_variable, topk  in itertools.product(color_grad_variable_vals, topk_vals):
			_plot_nsw_crossenc_recall_vs_comp_budget_old(
				all_results=all_results,
				plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}",
				model_type="crossenc",
				topk=topk,
				color_grad_variable=color_grad_variable,
				graph_type=graph_type,
				max_nbr_vals=max_nbr_vals,
				beamsize_vals=beamsize_vals,
				budget_vals=budget_vals,
			)
			# _plot_nsw_crossenc_recall_vs_comp_budget(
			# 	all_results=all_results,
			# 	plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}",
			# 	model_type="bienc",
			# 	topk=topk,
			# 	color_grad_variable=color_grad_variable,
			# 	graph_type=graph_type,
			# 	max_nbr_vals=max_nbr_vals,
			# 	beamsize_vals=beamsize_vals,
			# 	budget_vals=budget_vals
			# )
		
		# topk_vals = [64]
		# for color_grad_variable, topk  in itertools.product(color_grad_variable_vals, topk_vals):
		# 	_plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}", model_type="crossenc", topk=topk, metric="acc", color_grad_variable=color_grad_variable)
		# 	# # _plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact", model_type="bienc", topk=topk, metric="acc", color_grad_variable=color_grad_variable)
		#
		# 	_plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact{misc}", model_type="crossenc", topk=topk, metric="recall", color_grad_variable=color_grad_variable)
		# 	# # _plot_nsw_crossenc_metric_vs_comp_budget(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/{graph_type}_recall_wrt_exact", model_type="bienc", topk=topk, metric="recall", color_grad_variable=color_grad_variable)
		
		
	except Exception as e:
		embed()
		raise e
		

def _plot_search_comparison(all_results, embed_types, plt_dir):
	
	beamsize_vals = [1, 2, 5, 10, 20, 50, 100, 200]
	metrics = ["acc", "norm_acc", "recall", "num_score_comps~mean"]
	topk = 100
	
	X = np.arange(len(beamsize_vals))
	
	for metric in metrics:
		y_vals = []
		for _temp_et in embed_types:
			if "num_score_comps" in metric: continue
			y_vals += [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"crossenc~nsw~{metric}"]) for beamsize in beamsize_vals] + \
					  [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"crossenc~exact~{metric}"]) for beamsize in beamsize_vals] + \
					  [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"bienc~nsw~{metric}"]) for beamsize in beamsize_vals] + \
					  [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"bienc~exact~{metric}"]) for beamsize in beamsize_vals] + \
					  [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"bienc~nsw_reranked~{metric}"]) for beamsize in beamsize_vals] + \
					  [float(all_results[_temp_et][f"k={topk}_b={beamsize}_init_random"][f"bienc~exact_reranked~{metric}"]) for beamsize in beamsize_vals]
		
		for embed_type in embed_types:
			plt.clf()
			
			###################################### PLOTS WHEN INITIALIZING RANDOMLY ####################################
			if "num_score_comps" not in metric:
				Y_init = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"crossenc~init~{metric}"]) for beamsize in beamsize_vals]
				Y_cross_exact = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"crossenc~exact~{metric}"]) for beamsize in beamsize_vals]
				Y_bienc_exact = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"bienc~exact~{metric}"]) for beamsize in beamsize_vals]
				Y_bienc_exact_rerank = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"bienc~exact_reranked~{metric}"]) for beamsize in beamsize_vals]
				Y_bienc_nsw_rerank = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"bienc~nsw_reranked~{metric}"]) for beamsize in beamsize_vals]
			else:
				Y_init = []
				Y_cross_exact = []
				Y_bienc_exact = []
				Y_bienc_exact_rerank = [topk]*len(beamsize_vals)
				Y_bienc_nsw_rerank = [topk]*len(beamsize_vals)
			
			
			Y_cross_nsw = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"crossenc~nsw~{metric}"]) for beamsize in beamsize_vals]
			Y_bienc_nsw = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random"][f"bienc~nsw~{metric}"]) for beamsize in beamsize_vals]
			
			if Y_init: plt.plot(X, Y_init, "*-", c="grey", label="Random")
			
			plt.plot(X, Y_bienc_nsw, "*-", c="firebrick", label="BiEnc w/ NSW ")
			if Y_bienc_exact: plt.plot(X, Y_bienc_exact, "s-", c="firebrick", label="BiEnc (exact)")
			
			plt.plot(X, Y_bienc_nsw_rerank, "*-", c="deepskyblue", label="Bi-Enc w/ NSW -> Cross-Enc ")
			plt.plot(X, Y_bienc_exact_rerank, "s-", c="deepskyblue", label="Bi-E nc (exact) -> Cross-Enc")
			
			plt.plot(X, Y_cross_nsw, "*-", c="yellowgreen", label="Cross-Enc w/ NSW")
			if Y_cross_exact: plt.plot(X, Y_cross_exact, "s-", c="yellowgreen", label="Cross-Enc (exact)")
			############################################################################################################
			
			####################### PLOTS WHEN INITIALIZING USING NSW SEARCH WITH CORRESPONDING EMBEDDING TYPE #########
			if "num_score_comps" not in metric:
				Y_init = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}"][f"crossenc~init~{metric}"]) for beamsize in beamsize_vals]
				Y_bienc_nsw_rerank = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}"][f"bienc~nsw_reranked~{metric}"]) for beamsize in beamsize_vals]
			else:
				Y_init = []
				Y_bienc_nsw_rerank = [topk]*len(beamsize_vals)
			
			
			Y_cross_nsw = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}"][f"crossenc~nsw~{metric}"]) for beamsize in beamsize_vals]
			Y_bienc_nsw = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}"][f"bienc~nsw~{metric}"]) for beamsize in beamsize_vals]
			
			if Y_init: plt.plot(X, Y_init, "*--", c="grey", label="Random")
			plt.plot(X, Y_bienc_nsw, "*--", c="firebrick", label="BiEnc w/ NSW ")
			plt.plot(X, Y_bienc_nsw_rerank, "*--", c="deepskyblue", label="Bi-Enc w/ NSW -> Cross-Enc ")
			plt.plot(X, Y_cross_nsw, "*--", c="yellowgreen", label="Cross-Enc w/ NSW")
			############################################################################################################
			
			
			
			
			# if y_vals:
			# 	plt.ylim(min(y_vals) - 5, max(y_vals)+5)
			plt.xticks(X, beamsize_vals)
			plt.xlabel("Beam Size")
			plt.ylabel(metric)
			plt.title("Searching NSW using different types of scores")
			
			from matplotlib.lines import Line2D
			
			legend_elements = [Line2D([0], [0], ls='-', marker="s", color='k', label='Exact'),
							   Line2D([0], [0], ls='-', marker="*", color='k', label='NSW w/ random init'),
							   Line2D([0], [0], ls='--', marker="*", color='k', label='NSW w/ smart init'),
							   
							   Line2D([0], [0], ls='-', color='grey', label='Init'),
							   Line2D([0], [0], ls='-', color='firebrick', label='BiEnc'),
							   Line2D([0], [0], ls='-', color='deepskyblue', label='BiEnc -> CrossEnc'),
							   Line2D([0], [0], ls='-', color='yellowgreen', label='CrossEnc')]
			
			
			plt.legend(handles=legend_elements)
			# plt.legend()
			out_file = f"{plt_dir}/{embed_type}/{metric}_k={topk}.pdf"
			Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			plt.savefig(out_file)
			plt.close()
			
		
def _plot_graph_const_comparison(all_results, embed_types, plt_dir):
	# topk_vals = [10, 20, 50, 100]
	topk_vals = [100]
	beamsize_vals = [1, 2, 5, 10, 20, 50, 100, 200]
	metrics = list(all_results["tfidf"]["k=10_b=1"].keys())
	
	
	for topk in tqdm(topk_vals):
		X = np.arange(len(beamsize_vals))
		for metric in metrics:
			temp_keywords = ["exact", "recall_64", "recall_10", "recall_5", "mrr", "p1", "p10", "p90", "p99", "std"]
			skip_metric = False
			for _temp in temp_keywords:
				if _temp in metric:
					skip_metric = True
			if skip_metric: continue
			if "crossenc" in metric and "rerank" in metric: continue
			
			plt.clf()
			for embed_type in embed_types:
				Y = [float(all_results[embed_type][f"k={topk}_b={beamsize}"][metric]) for beamsize in beamsize_vals]
				plt.plot(X, Y, "*-", label=embed_type)
				
			plt.xticks(X, beamsize_vals)
			plt.xlabel("Beam Size")
			plt.ylabel(metric)
			plt.title(f"Metric = {metric} for top_k = {topk}")
			plt.legend()
			out_file = f"{plt_dir}/" + "/".join(metric.split("~")) + f"_k={topk}.pdf"
			Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			plt.savefig(out_file)
			plt.close()
	

def _plot_nsw_crossenc_recall_vs_comp_budget_old(all_results, plt_dir, model_type, topk, color_grad_variable, graph_type,
											 max_nbr_vals, beamsize_vals, budget_vals):
	
	try:
		embed_types = list(all_results.keys())
		budget_vals = [b for b in budget_vals if b is not None]
		
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]

		plt.clf()
		figure, axis = plt.subplots(1,2, figsize=(18,6))
		
		cmap_vals = ["Reds", "Greens", "Blues"]
		assert len(cmap_vals) >= len(embed_types)
		## Plot for bienc/tfidf initialization
		for embed_type, c, cmap in zip(embed_types, colors,  cmap_vals):
			entry_method = embed_type
			X, Y, color_vals = [], [], []
			retr_X, retr_Y = [], []
		
			for beamsize, max_nbrs in itertools.product(beamsize_vals, max_nbr_vals):
				retr_Y += [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{entry_method}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0][5:]) ]
				retr_X += [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{entry_method}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"] ]

				curr_X = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{entry_method}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"]
					  for budget in budget_vals]

				curr_Y = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{entry_method}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0]
					  for budget in budget_vals]
				curr_Y = [float(y[5:]) for y in curr_Y]

				X += curr_X
				Y += curr_Y
				if color_grad_variable == "max_nbrs":
					color_vals += [np.log(max_nbrs) if max_nbrs is not None else np.log(64)]*len(budget_vals)
				elif color_grad_variable == "beamsize":
					color_vals += [np.log(beamsize+1)] *len(budget_vals)
				else:
					color_vals += [c[1]]*len(budget_vals)
				
			edgecolors = [c[1]]*len(X)
			axis[0].scatter(X, Y, c=color_vals, edgecolors=edgecolors, alpha=0.8, cmap=plt.get_cmap(cmap))
			axis[0].scatter(retr_X, retr_Y, label=f"{graph_type}~{embed_type}~top", marker="x", c=c[0], alpha=0.8)
			temp ={x:y for x,y in zip(retr_X,retr_Y)}
			LOGGER.info(f"topk={topk}, embed_type: {embed_type} {temp} ")
		
		## Plot for random initialization
		for embed_type, c, cmap in zip(embed_types, colors, cmap_vals):
			X, Y, color_vals = [], [], []
			retr_X, retr_Y = [], []
			for beamsize, max_nbrs in itertools.product(beamsize_vals, max_nbr_vals):
				retr_Y += [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0][5:]) ]
				retr_X += [ all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"] ]

				curr_X = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~{graph_type}~num_score_comps~mean"]
					  for budget in budget_vals]

				curr_Y = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~exact_vs_{graph_type}~common_frac"][0]
					  for budget in budget_vals]
				curr_Y = [float(y[5:]) for y in curr_Y]

				X += curr_X
				Y += curr_Y
				if color_grad_variable == "max_nbrs":
					color_vals += [np.log(max_nbrs) if max_nbrs is not None else np.log(64)]*len(budget_vals)
				elif color_grad_variable == "beamsize":
					color_vals += [np.log(beamsize+1)]*len(budget_vals)
				else:
					color_vals += [c[1]]*len(budget_vals)
			
			edgecolors = [c[1]]*len(X)
			axis[1].scatter(X, Y, c=color_vals, edgecolors=edgecolors, alpha=0.8, cmap=plt.get_cmap(cmap))
			axis[1].scatter(retr_X, retr_Y, marker="x", label=f"{graph_type}~{embed_type}~random", c=c[0], alpha=0.8)
		
		
		
		[ax.set_ylim(0,1) for ax in axis]
		[ax.set_xlabel(f"Number of {model_type} calls") for ax in axis]
		axis[0].set_ylabel(f"Recall wrt exact {model_type} model")
		domain = all_results[embed_types[0]]["data_info"][0]
		try:
			n_ent = int(all_results[embed_types[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1][:-4])
		except ValueError:
			n_ent = int(all_results[embed_types[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1].split("_all_layers")[0])
			
		figure.suptitle(f"Recall wrt exact {model_type} for domain {domain} w/ {n_ent} entities")
		[ax.legend() for ax in axis]
		[ax.grid() for ax in axis]
		out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlinear{color_grad_variable}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		# plt.savefig(out_file)
		
		[ax.set_xscale("log") for ax in axis]
		out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlog{color_grad_variable}.pdf"
		plt.savefig(out_file)
		
		# [ax.set_xlim(0.1,1000) for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlog_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		#
		# [ax.set_xscale("linear") for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/recall_wrt_exact_vs_budget_xlinear_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		plt.close()
	except Exception as e:
		embed()
		raise e
		

def _plot_nsw_crossenc_metric_vs_comp_budget(all_results, embed_types, plt_dir, model_type, topk, metric, color_grad_variable):
	
	try:
		
		max_nbr_vals = [5, 10, 20, 50, None]
		beamsize_vals = [1, 2, 5, 10, 50, 100, 500, 1000]
		budget_vals = [None, 100, 500, 1000, 2000]
		
		max_nbr_vals = [5, 10, 20, 50, None]
		beamsize_vals = [1, 2, 5, 10, 50, 64, 100, 500, 1000]
		budget_vals = [None, 0, 64, 100, 250, 500, 1000, 2000]
		
		max_nbr_vals = [5, 10, 20, 50]
		topk_vals = [1, 10, 64, 100, 250, 1000]
		beamsize_vals = [1, 2, 5, 10, 50, 64, 100, 250, 500, 1000]
		budget_vals = [None, 0, 64, 100, 250, 500, 1000, 2000]
	
		
		
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]

		plt.clf()
		figure, axis = plt.subplots(1,2, figsize=(18,6))
		
		cmap_vals = ["Reds", "Greens"]
		# cmap_vals = ["autumn", "winter"]
		assert len(cmap_vals) >= len(embed_types)
		## Plot for top-k wrt embedding initialization
		for embed_type, c, cmap in zip(embed_types, colors, cmap_vals):
			X, Y, color_vals = [], [], []
			retr_X, retr_Y = [], []
			for max_nbrs in max_nbr_vals:
				for beamsize in beamsize_vals:
					retr_X += [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"] ]
					retr_Y += [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~{metric}"]) ]
					
					curr_X = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"]
						  for budget in budget_vals]
					
					curr_Y = [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~{metric}"])
						  for budget in budget_vals]
				
					X += curr_X
					Y += curr_Y
					if color_grad_variable == "max_nbrs":
						color_vals += [np.log(max_nbrs) if max_nbrs is not None else np.log(64)]*len(budget_vals)
					elif color_grad_variable == "beamsize":
						color_vals += [np.log(beamsize+1)]*len(budget_vals)
					else:
						color_vals += [c[1]]*len(budget_vals)
				
					# plt.scatter(curr_X, curr_Y, marker=marker ,c=c)
					# plt.scatter(curr_X, curr_Y, c=c, label=f"{embed_type}", alpha=0.6)
				 
				# plt.scatter(curr_X, curr_Y, marker=marker ,c=c, label=embed_type)
				exact_metric = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget=None_max_nbrs={max_nbrs}"][f"{model_type}~exact~{metric}"])]*len(X)
				axis[0].plot(X, exact_metric, "-", c=c[1])
				
			edgecolors = [c[1]]*len(X)
			axis[0].scatter(X, Y, c=color_vals, edgecolors=edgecolors, alpha=0.8, cmap=plt.get_cmap(cmap))
			axis[0].scatter(retr_X, retr_Y, marker="x", label=f"nsw~{embed_type}", c=c[0], alpha=0.8)
			
		
		## Plot for random initialization
		for embed_type, c, cmap in zip(embed_types, colors, cmap_vals):
			X, Y, color_vals = [], [], []
			retr_X, retr_Y = [], []
			for max_nbrs in max_nbr_vals:
				for beamsize in beamsize_vals:
					retr_X += [ all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"] ]
					retr_Y += [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~{metric}"]) ]
					
					curr_X = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"]
						  for budget in budget_vals]
					
					curr_Y = [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~{metric}"])
						  for budget in budget_vals]
				
					X += curr_X
					Y += curr_Y
					if color_grad_variable == "max_nbrs":
						color_vals += [np.log(max_nbrs) if max_nbrs is not None else np.log(64)]*len(budget_vals)
					elif color_grad_variable == "beamsize":
						color_vals += [np.log(beamsize+1)]*len(budget_vals)
					else:
						color_vals += [c[1]]*len(budget_vals)
				
					# plt.scatter(curr_X, curr_Y, marker=marker ,c=c)
					# plt.scatter(curr_X, curr_Y, c=c, label=f"{embed_type}", alpha=0.6)
				 
				# plt.scatter(curr_X, curr_Y, marker=marker ,c=c, label=embed_type)
				exact_metric = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_random_budget=None_max_nbrs={max_nbrs}"][f"{model_type}~exact~{metric}"])]*len(X)
				axis[1].plot(X, exact_metric, "-", c=c[1])
				
			edgecolors = [c[1]]*len(X)
			axis[1].scatter(X, Y, c=color_vals, edgecolors=edgecolors, alpha=0.8, cmap=plt.get_cmap(cmap))
			axis[1].scatter(retr_X, retr_Y, marker="x", label=f"nsw~{embed_type}", c=c[0], alpha=0.8)
		
		
		[ax.set_xlabel(f"Number of {model_type} calls") for ax in axis]
		axis[0].set_ylabel(f"{metric} of {model_type} model using NSW search")
		domain = all_results[embed_types[0]]["data_info"][0]
		try:
			n_ent = int(all_results[embed_types[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1][:-4])
		except ValueError:
			n_ent = int(all_results[embed_types[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1].split("_all_layers")[0])
		
		figure.suptitle(f"{metric} of {model_type} model using NSW search for domain {domain} w/ {n_ent} entities")
		[ax.legend() for ax in axis]
		[ax.grid() for ax in axis]
		out_file = f"{plt_dir}/{model_type}_k={topk}/{metric}_vs_budget_xslinear{color_grad_variable}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		# plt.savefig(out_file)
		
		[ax.set_xscale("log") for ax in axis]
		out_file = f"{plt_dir}/{model_type}_k={topk}/{metric}_vs_budget_xlog{color_grad_variable}.pdf"
		plt.savefig(out_file)
		
		# [ax.set_xlim(0.1,1000) for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/{metric}_vs_budget_xlog_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		#
		# [ax.set_xscale("linear") for ax in axis]
		# out_file = f"{plt_dir}/{model_type}_k={topk}/{metric}_vs_budget_xlinear_0-1K{color_grad_variable}.pdf"
		# plt.savefig(out_file)
		plt.close()
	except Exception as e:
		embed()
		raise e


def compare_nsw_vs_hnsw(data_info, res_dir):
	
	try:
		embed_types = ["tfidf", "bienc"]
		data_name, data_fnames = data_info
		plt_dir = f"{res_dir}/search_plots"
		Path(plt_dir).mkdir(exist_ok=True, parents=True)
		
		all_results = {}
		for embed_type in embed_types:
			try:
				filename = f"{res_dir}/nsw_eval_{data_name}_emb={embed_type}.json"
				with open(filename, "r") as fin:
					all_results[embed_type] = json.load(fin)
			except:
				LOGGER.info(f"File not found : {filename}")
		
		
		model_type = "crossenc"
		rand_conditions = [lambda x: "init_random" not in x, lambda x: "init_random" in x ]
		all_ans = {"files": [f"{res_dir}/nsw_eval_{data_name}_emb={embed_type}.json" for embed_type in embed_types],
				   "metric": ["Recall fraction wrt exact crossencoder model : NSW - HNSW, x: recall diff, y: num crossenc call diff"]}
		for embed_type, rand_condition in itertools.product(all_results, rand_conditions):
			y_diffs, x_diffs = [], []
			for key in all_results[embed_type]:
				if key in ["data_info", "bi_model_config"]: continue
				if rand_condition(key): continue
				y_hnsw = float(all_results[embed_type][key][f"{model_type}~exact_vs_hnsw~common_frac"][0][5:])
				y_nsw = float(all_results[embed_type][key][f"{model_type}~exact_vs_nsw~common_frac"][0][5:])
				x_hnsw  = all_results[embed_type][key][f"{model_type}~hnsw~num_score_comps~mean"]
				x_nsw  = all_results[embed_type][key][f"{model_type}~nsw~num_score_comps~mean"]
				
				y_diffs += [100*(y_nsw - y_hnsw)]
				x_diffs += [x_nsw - x_hnsw]
			
			ans = {
				"Mean": {"x":f"{np.mean(y_diffs):.2f}", "y":f"{np.mean(x_diffs):.2f}"},
				"Min": {"x":f"{np.min(y_diffs):.2f}", "y":f"{np.min(x_diffs):.2f}"},
				"p10": {"x":f"{np.percentile(y_diffs, 10):.2f}", "y":f"{np.percentile(x_diffs, 10):.2f}"},
				"p50": {"x":f"{np.percentile(y_diffs, 50):.2f}", "y":f"{np.percentile(x_diffs, 50):.2f}"},
				"p90": {"x":f"{np.percentile(y_diffs, 90):.2f}", "y":f"{np.percentile(x_diffs, 90):.2f}"},
				"Max": {"x":f"{np.max(y_diffs):.2f}", "y":f"{np.max(x_diffs):.2f}"}
			}
			all_ans[f"embed_type={embed_type}_init_random={not rand_condition(key)}"] = ans
		
		
		LOGGER.info(json.dumps(all_ans, indent=4))
		with open(f"{res_dir}/nsw_vs_hsnw.json", "w") as fout:
			json.dump(all_ans, fout, indent=4)
		LOGGER.info("Done")
		# embed()
	except Exception as e:
		embed()
		raise e
	
	
def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	parser.add_argument("--project_name", type=str, default="NSW-Eval", help="WANDB project name")
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions to run NSW inference on. -1 for all ments")
	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor", "none", "c-anchor"], required=True, help="Type of embeddings to use for building NSW")
	
	
	parser.add_argument("--bi_model_file", type=str, default="", help="Biencoder Model config file or ckpt file")
	parser.add_argument("--e2e_score_filename", type=str, default="", help="Pickle file containing entity-entity scores information")
	parser.add_argument("--a2e_score_filename", type=str, default="", help="Pickle file containing anchor-entity scores information (this is used to get entity embeddings for indexing entities)")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--score_mat_dir", type=str, default="", help="Dir with precomputed score mats. If not specified then it is assigned value of res_dir")
	parser.add_argument("--graph_type", type=str, default="nsw", choices=["knn", "nsw", "hnsw", "knn_e2e", "nsw_e2e", "rand"], help="Type of graph to use")
	parser.add_argument("--entry_method", type=str, default="bienc", choices=["bienc", "tfidf", "random"], help="Method for choosing entry points in graph")
	parser.add_argument("--force_exact_init_search", type=int, default=0, choices=[0, 1], help="Force exact search when finding initial entry points")
	parser.add_argument("--graph_metric", type=str, default="l2", choices=["l2", "ip"], help="Metric/distance to use for building NSW")
	parser.add_argument("--plot_only", type=int, default=0, choices=[0, 1], help="1 to only plot results and 0 to run (h)nsw search and then plot")
	parser.add_argument("--misc", type=str, default="", help="misc suffix to add to result file")
	parser.add_argument("--debug_mode", type=int, default=0, help="run in debug mode w/ wandb logging turned off")
	
	args = parser.parse_args()
	project_name = args.project_name
	data_name = args.data_name
	n_ment = args.n_ment
	embed_type = args.embed_type
	
	bi_model_file = args.bi_model_file
	res_dir = args.res_dir
	score_mat_dir = args.score_mat_dir
	score_mat_dir = res_dir if score_mat_dir == "" else score_mat_dir # If score_mat_dir value is not given then use res_dir
	
	
	graph_type = args.graph_type
	entry_method = args.entry_method
	force_exact_init_search = bool(args.force_exact_init_search)
	graph_metric = args.graph_metric
	e2e_score_filename = args.e2e_score_filename
	a2e_score_filename = args.a2e_score_filename
	
	plot_only = bool(args.plot_only)
	debug_mode = bool(args.debug_mode)
	misc = "_" + args.misc if args.misc != "" else ""
	
	n_ment = n_ment if n_ment != -1 else None
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=score_mat_dir, worlds=worlds, n_ment=n_ment)
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	assert embed_type != "bienc" or bi_model_file != "", f"bi_model_file should be not empty if embed_type == {embed_type}"
	iter_worlds = worlds[:4] if data_name == "all" else [("", data_name)]
	
	config={
			"goal": "Run NSW Eval",
			"data_name":data_name,
			"embed_type":embed_type,
			"bi_model_config":bi_model_file,
			"matrix_dir":res_dir,
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(args.__dict__)

	if False and debug_mode:
		# For quick debugging
		max_nbr_vals = [10]
		topk_vals = [1, 10, 64]
		beamsize_vals = [1, 10, 64]
		budget_vals = [None, 0, 1, 10, 64]
	else:
		max_nbr_vals = [5, 10, 20, 50]
		topk_vals = [1, 10, 64, 100, 250, 500, 1000]
		beamsize_vals = [1, 2, 5, 10, 50, 64, 100, 250, 500, 1000]
		budget_vals = [None, 0, 64, 100, 250, 500, 1000, 2000]
	
	
	config["max_nbr_vals"] = max_nbr_vals
	config["topk_vals"] = topk_vals
	config["beamsize_vals"] = beamsize_vals
	config["budget_vals"] = budget_vals
	command = ' '.join([str(x) for x in sys.argv])
	config["command"] = command
	LOGGER.info(f"Command : {command}")
	wandb.init(
		project=project_name,
		dir="../../results/6_ReprCrossEnc/PooledResults",
		config=config,
		mode="disabled" if debug_mode else "online"
	)
	
	wandb.run.summary["status"] = 0
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		if not plot_only:
			run(
				res_dir=f"{res_dir}/{world_name}/{graph_type}",
				data_info=(world_name, DATASETS[world_name]),
				embed_type=embed_type,
				bi_model_file=bi_model_file,
				graph_metric=graph_metric,
				graph_type=graph_type,
				entry_method=entry_method,
				force_exact_init_search=force_exact_init_search,
				misc=misc,
				max_nbr_vals=max_nbr_vals,
				topk_vals=topk_vals,
				beamsize_vals=beamsize_vals,
				budget_vals=budget_vals,
				e2e_score_filename=e2e_score_filename,
				a2e_score_filename=a2e_score_filename,
				arg_dict=args.__dict__
			)
		
		wandb.run.summary["status"] = 6
		if plot_only:
			plot(
				res_dir=f"{res_dir}/{world_name}/{graph_type}",
				data_info=(world_name, DATASETS[world_name]),
				misc=misc,
				max_nbr_vals=max_nbr_vals,
				topk_vals=topk_vals,
				beamsize_vals=beamsize_vals,
				budget_vals=budget_vals,
				graph_type=graph_type,
				graph_metric=graph_metric,
			)
		
		# compare_nsw_vs_hnsw(
		# 	res_dir=f"{res_dir}/{world_name}/{graph_type}",
		# 	data_info=(world_name, DATASETS[world_name])
		# )
		# wandb.run.summary["status"] = 7


if __name__ == "__main__":
	main()

