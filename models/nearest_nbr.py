import sys
import math
import json
import torch
import faiss
import pickle
import logging
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from IPython import embed
from abc import ABCMeta, abstractmethod

from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import normalize

from eval.eval_utils import compute_label_embeddings

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


class HNSWWrapper(object):
	
	def __init__(self, dim, data, max_nbrs, metric):
		"""
		
		:param dim:
		:param data:
		"""
		self.metric = metric
		self.dim = dim
		self.max_nbrs = max_nbrs
		
		
		
		# FAISS HNSW file : https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h /HNSW.cpp
		self.index = self.build_index(
			dim=self.dim,
			data=data,
			max_nbrs=self.max_nbrs,
			metric=self.metric
		)
		
	@staticmethod
	def build_index(dim, data, max_nbrs, metric):
		
		LOGGER.info(f"Initializing index")
		if metric == "ip": # Inner-product
			# Add 1 extra dim to as we will modify data to support max inner product search
			index = faiss.index_factory(dim+1, "HNSW32") # 32 here signifies that we will index float32 vectors
		else:
			index = faiss.index_factory(dim, "HNSW32") # 32 here signifies that we will index float32 vectors
		
		n_levels = len(HNSWWrapper.vector_to_array(index.hnsw.cum_nneighbor_per_level)) - 1
		
		# LOGGER.info(f"(Default max nbrs) :{[(level,index.hnsw.nb_neighbors(level)) for level in range(n_levels)]}")
		if max_nbrs is not None:
			if isinstance(max_nbrs, int):
				temp  = [index.hnsw.set_nb_neighbors(level, max_nbrs) for level in range(n_levels)]
			else:
				assert len(max_nbrs) == n_levels, f"Max nbr array len = {len(max_nbrs)} not same as level array {n_levels}"
				temp  = [index.hnsw.set_nb_neighbors(level, curr_max_nbrs) for level, curr_max_nbrs in zip(range(n_levels), max_nbrs)]
		
		# LOGGER.info(f"(After setting max nbrs) :{[(level, index.hnsw.nb_neighbors(level)) for level in range(n_levels)]}")
		
		if data is not None and len(data) > 0:
			if metric == "ip": # Inner-product
				LOGGER.info(f"Adding extra dimension to better support metric = {metric}")
				data = HNSWWrapper.augment_xb(xb=data)
				
			index.add(data)
		
		LOGGER.info(f"Finished adding data to index")
		return index
	
	# @staticmethod
	# def get_max_norm(xb):
	# 	return (xb ** 2).sum(1).max()
	
	@staticmethod
	def augment_xb(xb, max_norm=None):
		try:
			norms = np.linalg.norm(xb,axis=1)
			norms = norms**2
			if max_norm is None:
				max_norm = norms.max()
			extracol = np.sqrt(max_norm - norms)
			return np.hstack((xb, extracol.reshape(-1, 1)))
		except Exception as e:
			LOGGER.info("Error in HNSW")
			embed()
			raise e
	
	@staticmethod
	def augment_x_query(xq):
		extracol = np.zeros(len(xq), dtype='float32')
		return np.hstack((xq, extracol.reshape(-1, 1)))
		
		
	@staticmethod
	def vector_to_array(v):
		""" make a vector visible as a numpy array (without copying data)"""
		return faiss.rev_swig_ptr(v.data(), v.size())
	
	def deserialize_from(self, index_path):
		self.index = faiss.read_index(index_path)
		# self.index.deserialize_from(index_path)
	
	
	def serialize(self, index_path):
		faiss.write_index(self.index, index_path)
		# self.index.serialize(index_path)
		
	@property
	def num_elems(self):
		return len(faiss.vector_to_array(self.index.hnsw.levels))
	
	
	def add_data(self, data):
		self.index.add(data)
	
	
	def get_nsw_graph_at_level(self, level):
		"""
		Returns dictionary mapping node_index to list containing its neighbours.
		:param level: int - level for which we need to extract NSW graph out of HNSW. Starts indexing from 1 for lowest level graph.
		:return:
		"""
		assert level > 0
		
		levels = self.get_levels() # Max level for each indexed vector
		
		# number of neighbors stored per layer (cumulative) - This is a constant for the index and is not specific to any node
		cum_nneighbor_per_level = self._get_cum_nneighbor_per_level()
		
		offsets = self._get_offsets() # Start and end offset of neighbors of each indexed vector in hnsw.neighbors
		neighbors = self._get_neighbors() # List of neighbors of all indexed vectors, across all levels of HNSW
		
		nsw_graph = {}
		for curr_idx in range(self.num_elems):
			if levels[curr_idx] < level:
				nsw_graph[curr_idx] = []
			else:
				# Get nbrs of current node across all levels of HNSW
				curr_nbrs = neighbors[offsets[curr_idx]: offsets[curr_idx + 1]]
				
				# Extract nbrs of current node at given level
				curr_level_nbrs  = curr_nbrs[cum_nneighbor_per_level[level - 1] : cum_nneighbor_per_level[level]]
				
				# Remove -1s from nbr list as they are used for padding nbr list
				curr_level_nbrs = curr_level_nbrs[curr_level_nbrs != -1]
				nsw_graph[curr_idx] = curr_level_nbrs
		
		return nsw_graph
		
	def get_levels(self):
		"""
		:return: array containing max_level for each indexed vector
		"""
		return self.vector_to_array(self.index.hnsw.levels) # Max level for each indexed vector
	
	def _get_cum_nneighbor_per_level(self):
		# number of neighbors stored per layer (cumulative) - This is a constant for the index and is not specific to any node
		return self.vector_to_array(self.index.hnsw.cum_nneighbor_per_level)
	
	def _get_offsets(self):
		return self.vector_to_array(self.index.hnsw.offsets) # Start and end offset of neighbors of each indexed vector in hnsw.neighbors

	def _get_neighbors(self):
		return self.vector_to_array(self.index.hnsw.neighbors) # List of neighbors of all indexed vectors, across all levels of HNSW
		
	def search(self, query_embed, k):
		if self.metric == "ip": # Inner-product
			query_embed = HNSWWrapper.augment_x_query(query_embed)
			
		return self.index.search(query_embed, k)


class TreeNode(object):
	
	def __init__(self, data_idxs=[], data_embeds=[]):
		self.left_child = None
		self.right_child = None
		self.pivot_idx = None
		self.pivot_embed = None
		
		assert len(data_idxs) == len(data_embeds), f"Len of idxs = {len(data_idxs)} does not match len of data = {len(data_embeds)}"
		self.data_idxs = data_idxs
		self.data_embeds = data_embeds
		
	
	@property
	def is_leaf(self):
		return self.left_child is None and self.right_child is None
	
 
class PivotTreeIndex(metaclass=ABCMeta):
	
	def __init__(self, embeds, max_samples_per_leaf):
		self.max_samples_per_leaf = max_samples_per_leaf
		self.rng = np.random.default_rng(seed=0) # TODO: How should this seed be chosen?
		self.index_root = self.build_index(embeds=embeds)
		
	def build_index(self, embeds):
		"""
		
		:param embeds: List or array of embeddings of items to index
		:return:
		"""
		n_elems = len(embeds)
		index = self._build_index(item_idxs=np.arange(n_elems), item_embeds=embeds)
		return index
	
	def _build_index(self, item_idxs, item_embeds):
		"""
		
		:param item_idxs: List of idxs of items
		:param item_embeds: List of embeddings of items
		:return:
		"""
		try:
			n_elems =  len(item_embeds)
			assert len(item_idxs) == len(item_embeds), f"Idxs len = {len(item_idxs)} and embed len = {len(item_embeds)} should be same."
			if n_elems <= self.max_samples_per_leaf: # Create leaf node
				return TreeNode(data_idxs=item_idxs, data_embeds=item_embeds)
			else: # Create left and right child of this node
				curr_node = TreeNode()
				# Choose a left and right pivot
				pivot_idxs = self.find_pivot_idxs(n_elems=n_elems, item_embeds=item_embeds)
				left_pivot_idx, right_pivot_idx = pivot_idxs[0], pivot_idxs[1]
				
				sim_w_left_pivot = np.dot(item_embeds[left_pivot_idx],item_embeds.T)
				sim_w_right_pivot = np.dot(item_embeds[right_pivot_idx],item_embeds.T)
				
				left_partition = np.squeeze(np.asarray(sim_w_left_pivot > sim_w_right_pivot))
				right_partition = np.squeeze(np.asarray(sim_w_left_pivot <= sim_w_right_pivot))
				
				left_child = self._build_index(item_idxs=item_idxs[left_partition], item_embeds=item_embeds[left_partition])
				right_child = self._build_index(item_idxs=item_idxs[right_partition], item_embeds=item_embeds[right_partition])
				
				left_child.pivot_idx = item_idxs[left_pivot_idx]
				left_child.pivot_embed = item_embeds[left_pivot_idx]
				
				right_child.pivot_idx = item_idxs[right_pivot_idx]
				right_child.pivot_embed = item_embeds[right_pivot_idx]
				
				curr_node.left_child = left_child
				curr_node.right_child = right_child
				
				return curr_node
		except Exception as e:
			embed()
			raise e
	
	@abstractmethod
	def find_pivot_idxs(self, n_elems, item_embeds):
		raise NotImplementedError
	
	def search(self, query_embed, beam_size, top_k, item_scores):
		"""
		
		:param query_embed:
		:param beam_size:
		:param top_k:
		:param item_scores:
		:return: List of idx, score tuples and number of items scored during search
		"""
		try:
			score_cache = {}
			
			# 1. Find best beam_size number of leaf nodes
			leaf_nodes = self._find_leaf_nodes(query_embed=query_embed, curr_nodes=[self.index_root], beam_size=beam_size,
											   score_cache=score_cache, item_scores=item_scores)
			
			# 2. Exhaustively search leaf nodes
			all_items_in_leaves = [idx for n in leaf_nodes for idx in n.data_idxs]
			leaf_item_scores = [item_scores[idx] for idx in all_items_in_leaves] # TODO: Replace with actual score computation
			
			score_cache.update({idx:score for idx, score in zip(all_items_in_leaves, leaf_item_scores)})
			
			items_and_scores = sorted(list(score_cache.items()), key=lambda x:x[1], reverse=True)
			top_k_items_scores = items_and_scores[:top_k]
			
			if len(top_k_items_scores) < top_k:
				top_k_items_scores += [top_k_items_scores[-1]]*top_k # Repeat last element to fill top-k spots
				top_k_items_scores = top_k_items_scores[:top_k]
			assert len(top_k_items_scores) == top_k
			return top_k_items_scores, len(score_cache)
			
		except Exception as e:
			embed()
			raise e
			
	
	def _find_leaf_nodes(self, query_embed, curr_nodes, beam_size, score_cache, item_scores):
		
		try:
			assert len(curr_nodes) <= beam_size, f"Beam contains {len(curr_nodes)} nodes which is greater than beam_size param = {len(beam_size)}"
			
			
			# TODO: 1) Cache scores against all internal nodes that are visited as internal nodes also store data
			num_leaves = sum(1 if curr_node.is_leaf else 0 for curr_node in curr_nodes)
			
			leaf_nodes = [curr_node for curr_node in curr_nodes if curr_node.is_leaf]
			if num_leaves == beam_size: # All leaf
				return leaf_nodes
			elif num_leaves > beam_size: # All leaf
				warnings.warn(f"Number of leaf nodes = {num_leaves} is greater than beam_size={beam_size}. We are using just first {beam_size} nodes")
				return leaf_nodes[:beam_size]
			else: # Perform beam search on non-leaf nodes
				
				# We have found num_leaves out beam_size nodes already, so reducing beam_size
				curr_beam_size = beam_size - num_leaves
				
				# TODO: Implement some logic to make sure that we are not just using first beam_size leaf nodes that we find, and that we can prune away some leaf nodes if we are at some other better internal node in the search process
				non_leaf_nodes = [curr_node for curr_node in curr_nodes if not curr_node.is_leaf]
				
				child_nodes = [[curr_node.left_child, curr_node.right_child] for curr_node in non_leaf_nodes]
				child_nodes = [n for ns in child_nodes for n in ns]
				
				idx_to_node = {n.pivot_idx:n for n in child_nodes}
				child_pivot_idxs = [n.pivot_idx for n in child_nodes]
				child_pivot_scores = [item_scores[idx] for idx in child_pivot_idxs] # TODO: Replace this with actual score computation
				
				score_cache.update({idx:score for idx,score in zip(child_pivot_idxs, child_pivot_scores)})
				
				
				idx_and_scores = sorted(list(zip(child_pivot_idxs, child_pivot_scores)), key=lambda x: x[1], reverse=True)
				
				top_k_idx_and_scores = idx_and_scores[:curr_beam_size]
				top_k_child_nodes = [idx_to_node[idx] for idx,_ in top_k_idx_and_scores]
				
				new_leaf_nodes = self._find_leaf_nodes(curr_nodes=top_k_child_nodes, query_embed=query_embed,
													   beam_size=curr_beam_size, score_cache=score_cache, item_scores=item_scores)
				
				all_leaf_nodes = leaf_nodes + new_leaf_nodes
				idx_to_node = {n.pivot_idx:n for n in all_leaf_nodes}
				
				all_leaf_nodes_idxs = [n.pivot_idx for n in all_leaf_nodes]
				all_leaf_nodes_idxs_and_scores = [(idx, score_cache[idx]) for idx in all_leaf_nodes_idxs]
				
				# Take top beam_size number of leaf nodes
				all_leaf_nodes_idxs_and_scores = sorted(all_leaf_nodes_idxs_and_scores, key=lambda x: x[1], reverse=True)[:beam_size]
				
				# Return chosen top-beam_size leaf nodes
				return [idx_to_node[idx] for idx,_ in all_leaf_nodes_idxs_and_scores]
				
		except Exception as e:
			embed()
			raise e
			
			
		
			
class RandPivotTreeIndex(PivotTreeIndex):
	"""
	PivotTreeIndex with pivots chosen uniformly at random
	"""
	def __init__(self, embeds, max_samples_per_leaf):
		super(RandPivotTreeIndex, self).__init__(embeds=embeds, max_samples_per_leaf=max_samples_per_leaf)
	
	def find_pivot_idxs(self, n_elems, item_embeds):
		return self.rng.choice(n_elems, size=2, replace=False)
		

class KCenterPivotTreeIndex(PivotTreeIndex):
	"""
	PivotTreeIndex with pivots chosen using k-center clustering
	"""
	def __init__(self, embeds, max_samples_per_leaf):
		super(KCenterPivotTreeIndex, self).__init__(embeds=embeds, max_samples_per_leaf=max_samples_per_leaf)
	
	def find_pivot_idxs(self, n_elems, item_embeds):
		# Implement K-center finding algorithm here
		raise NotImplementedError
	
	# lass sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
	

class KMeansPlusPivotTreeIndex(PivotTreeIndex):
	"""
	PivotTreeIndex with pivots chosen using k-means++ seeding algorithm
	"""
	def __init__(self, embeds, max_samples_per_leaf):
		super(KMeansPlusPivotTreeIndex, self).__init__(embeds=embeds, max_samples_per_leaf=max_samples_per_leaf)
	
	def find_pivot_idxs(self, n_elems, item_embeds):
		# Implement K-center finding algorithm here
		centers, indices = kmeans_plusplus(item_embeds, n_clusters=2, random_state=0)
		return indices
		
		
class KNNGraph(object):
	
	def __init__(self, data, max_nbrs, metric):
		self.knn_graph = self.build_index(
			data=data,
			max_nbrs=max_nbrs,
			metric=metric
		)
		self.num_elems = len(data)
		self.metric = metric
		
		
	@staticmethod
	def build_index(data, max_nbrs, metric):
		#TODO: See if we want to only return mutual nearest nbrs?
		
		if metric == "ip":
			# Add data to some nearest nbr search index
			index = build_flat_or_ivff_index(
					embeds=data,
					force_exact_search=False
				)
		elif metric == "l2":
			d = data.shape[1] # Dimension of vectors to embed
			index = faiss.IndexFlatL2(d)
			index.add(data)
		else:
			raise NotImplementedError(f"metric = {metric} not supported")
		

		# For each datapoint, find top-k nearest nbrs and return a kNN graph based on that
		kn_nbrs_scores, kn_nbrs_idxs = index.search(data, k=max_nbrs+1)
		
		assert kn_nbrs_idxs.shape == (len(data), max_nbrs+1), f"kn_nbrs_idxs.shape = {kn_nbrs_idxs.shape} != (len(data), max_nbrs+1) = {(len(data), max_nbrs+1)}"
		
		# FIXME: This is not entirely correct. This assumes that is a datapoint has highest similarity with itself but that
		# might be violated in case of inner-product. For <x, y> can be greater than <x,x> if y = 2*x.
		# Assuming each vector is most similar to itself so removing first colum in these
		kn_nbrs_scores, kn_nbrs_idxs = kn_nbrs_scores[:, 1:], kn_nbrs_idxs[:, 1:]
		
		graph = {}
		n_elems = len(kn_nbrs_idxs)
		for elem_id in range(n_elems):
			graph[elem_id] = kn_nbrs_idxs[elem_id]
		
		return graph
	
	
	
	def get_nsw_graph_at_level(self, level):
		"""
		Returns dictionary mapping node_index to list containing its neighbours.
		:param level: int - level for which we need to extract NSW graph out of HNSW. Starts indexing from 1 for lowest level graph.
		:return:
		"""
		assert level > 0
		return self.knn_graph
		
		
	def get_levels(self):
		"""
		:return: array containing max_level for each indexed vector
		"""
		return [1 for _ in range(self.num_elems)]


class RandomGraph(object):
	
	def __init__(self, n_elems, max_nbrs):
		self.knn_graph = self.build_index(
			n_elems=n_elems,
			max_nbrs=max_nbrs,
		)
		self.num_elems = n_elems
		
		
	@staticmethod
	def build_index(n_elems, max_nbrs):
		
		rng = np.random.default_rng(seed=0)
		
		graph = {}
		all_elem_ids = list(range(n_elems))
		for elem_id in range(n_elems):
			graph[elem_id] = rng.choice(all_elem_ids, replace=False, size=max_nbrs)
		
		return graph
	
	
	
	def get_nsw_graph_at_level(self, level):
		"""
		Returns dictionary mapping node_index to list containing its neighbours.
		:param level: int - level for which we need to extract NSW graph out of HNSW. Starts indexing from 1 for lowest level graph.
		:return:
		"""
		assert level > 0
		return self.knn_graph
		
		
	def get_levels(self):
		"""
		:return: array containing max_level for each indexed vector
		"""
		return [1 for _ in range(self.num_elems)]


class KNNGraphwEnt2Ent(object):
	
	def __init__(self, max_nbrs, e2e_score_filename):
		self.knn_graph = self.build_index(
			e2e_score_filename=e2e_score_filename,
			max_nbrs=max_nbrs,
		)
		self.num_elems = len(self.knn_graph)
		
		
	@staticmethod
	def build_index(e2e_score_filename, max_nbrs):
		#TODO: See if we want to only return mutual nearest nbrs?
		
		with open(e2e_score_filename, "rb") as fin:
			res = pickle.load(fin)
			
		ent_to_ent_scores = res["ent_to_ent_scores"]
		topk_ents = res["topk_ents"]
		n_ent_x = res["n_ent_x"]
		n_ent_y = res["n_ent_y"]
		token_opt = res["token_opt"]
		entity_id_list_x = res["entity_id_list_x"]
		entity_id_list_y = res["entity_id_list_y"]
		tokenized_entities = res["entity_tokens_list"]
		arg_dict = res["arg_dict"]

		graph = {}
		for ent_id in range(n_ent_x):
			curr_nbr_scores = ent_to_ent_scores[ent_id]
			curr_nbr_idxs  = topk_ents[ent_id]
			
			# Find top-k = max_nbrs + 1 ents. Then filter out ent_id from this list to remove a self-loop in the graph
			topk_scores, topk_temp_ids = torch.topk(torch.tensor(curr_nbr_scores), k=max_nbrs+1)
			topk_nbr_ids = curr_nbr_idxs[topk_temp_ids]
			
			topk_nbr_ids = [nbr_id for nbr_id in topk_nbr_ids if nbr_id != ent_id]
			topk_nbr_ids = topk_nbr_ids[:max_nbrs]
			assert len(topk_nbr_ids) == max_nbrs, f"len(topk_nbr_ids) = {len(topk_nbr_ids)} != max_nbrs = {max_nbrs}"
			graph[ent_id] = topk_nbr_ids
		
		
		return graph
		
		
	
	def get_nsw_graph_at_level(self, level):
		"""
		Returns dictionary mapping node_index to list containing its neighbours.
		:param level: int - level for which we need to extract NSW graph out of HNSW. Starts indexing from 1 for lowest level graph.
		:return:
		"""
		assert level > 0
		return self.knn_graph
		
		
	def get_levels(self):
		"""
		:return: array containing max_level for each indexed vector
		"""
		return [1 for _ in range(self.num_elems)]
		

class NSWGraphwEnt2Ent(object):
	
	def __init__(self, max_nbrs, e2e_score_filename, n_long_range):
		self.nsw_graph = self.build_index(
			e2e_score_filename=e2e_score_filename,
			max_nbrs=max_nbrs,
			n_long_range=n_long_range
		)
		self.num_elems = len(self.nsw_graph)
		
		
	@staticmethod
	def build_index(e2e_score_filename, max_nbrs, n_long_range):
		
		
		try:
			
			with open(e2e_score_filename, "rb") as fin:
				res = pickle.load(fin)
				
			ent_to_ent_scores = res["ent_to_ent_scores"]
			topk_ents = res["topk_ents"]
			n_ent_x = res["n_ent_x"]
			n_ent_y = res["n_ent_y"]
			token_opt = res["token_opt"]
			entity_id_list_x = res["entity_id_list_x"]
			entity_id_list_y = res["entity_id_list_y"]
			tokenized_entities = res["entity_tokens_list"]
			arg_dict = res["arg_dict"]
		
			rng = np.random.default_rng(seed=0)
			
			graph = {}
			for ent_id in range(n_ent_x):
				curr_nbr_scores = ent_to_ent_scores[ent_id]
				curr_nbr_idxs  = topk_ents[ent_id]
				
				# Find top-k = max_nbrs + 1 ents. Then filter out ent_id from this list to remove a self-loop in the graph
				topk_scores, topk_temp_ids = torch.topk(torch.tensor(curr_nbr_scores), k=max_nbrs+1)
				topk_nbr_ids = curr_nbr_idxs[topk_temp_ids]
				
				topk_nbr_ids = [nbr_id for nbr_id in topk_nbr_ids if nbr_id != ent_id]
				topk_nbr_ids = topk_nbr_ids[:max_nbrs]
				
				long_range_nbrs = []
				while len(long_range_nbrs) < n_long_range:
					long_range_nbrs = [nbr_id for nbr_id in rng.integers(n_ent_y, size=10*n_long_range) if nbr_id not in set(topk_nbr_ids) and nbr_id != ent_id]
					
				
				final_nbrs = topk_nbr_ids[:max_nbrs - n_long_range] + long_range_nbrs[:n_long_range]
				final_nbrs = topk_nbr_ids[:max_nbrs - n_long_range]
				
				# assert len(final_nbrs) == max_nbrs, f"len(topk_nbr_ids) = {len(final_nbrs)} != max_nbrs = {max_nbrs}"
				# graph[ent_id] = topk_nbr_ids # FIXME: Use final_nbrs here instead of topk_nbr_ids
				graph[ent_id] = final_nbrs
			
			return graph
		except Exception as e:
			LOGGER.info(f"Exception in build_index for KNNGraphwEnt2Ent = {e}")
			embed()
			raise e
			
		
	
	def get_nsw_graph_at_level(self, level):
		"""
		Returns dictionary mapping node_index to list containing its neighbours.
		:param level: int - level for which we need to extract NSW graph out of HNSW. Starts indexing from 1 for lowest level graph.
		:return:
		"""
		assert level > 0
		return self.nsw_graph
		
		
	def get_levels(self):
		"""
		:return: array containing max_level for each indexed vector
		"""
		return [1 for _ in range(self.num_elems)]

	

def build_flat_or_ivff_index(embeds, force_exact_search, probe_mult_factor=1):
	
	LOGGER.info(f"Beginning indexing given {len(embeds)} embeddings")
	if type(embeds) is not np.ndarray:
		if torch.is_tensor(embeds):
			embeds = embeds.numpy()
		else:
			embeds = np.array(embeds)
	
	# Build index
	d = embeds.shape[1] # Dimension of vectors to embed
	nembeds = embeds.shape[0] # Number of elements to embed
	if nembeds <= 11000 or force_exact_search:  # if the number of embeddings is small, don't approximate
		index = faiss.IndexFlatIP(d)
		index.add(embeds)
	else:
		# number of quantized cells
		nlist = int(math.floor(math.sqrt(nembeds)))
		
		# number of the quantized cells to probe
		nprobe = int(math.floor(math.sqrt(nlist) * probe_mult_factor))
		
		quantizer = faiss.IndexFlatIP(d)
		index = faiss.IndexIVFFlat(
			quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
		)
		index.train(embeds)
		index.add(embeds)
		index.nprobe = nprobe
	
	LOGGER.info("Finished indexing given embeddings")
	return index


def embed_tokenized_entities(biencoder, ent_tokens_file):
	
	complete_entity_tokens_list = np.load(ent_tokens_file)
	
	# complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list).to(biencoder.device)
	complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
	
	embeds = compute_label_embeddings(biencoder=biencoder,
									  labels_tokens_list=complete_entity_tokens_list,
									  batch_size=50)
	
	return embeds


def index_tokenized_entities(biencoder, ent_tokens_file, force_exact_search):
	
	embeds = embed_tokenized_entities(biencoder=biencoder,
							 ent_tokens_file=ent_tokens_file)
	
	index = build_flat_or_ivff_index(embeds=embeds,
									 force_exact_search=force_exact_search)
	
	return index, embeds


def temp_search(embeds, index):
	try:
		k = 10
		for idx, curr_embed in enumerate(tqdm(embeds)):
			curr_embed = curr_embed.reshape(1,-1)
			nn_ent_dists, nn_ent_idxs = index.search(curr_embed, k)
			embed()
			input("")
			# if idx > 10:
			# 	break
	except Exception as e:
		embed()
		raise e


def main():
	
	parser = argparse.ArgumentParser( description='Build Nearest Nbr Index')
	
	parser.add_argument("--ent_file", type=str, required=True, help="File containing tokenized entities or entity embeddings")
	# parser.add_argument("--out_file", type=str, required=True, help="Output file storing index")
	
	parser.add_argument("--model_config", type=str, default=None, help="Model config for embedding model")
	
	
	args = parser.parse_args()
	
	ent_file = args.ent_file
	model_config = args.model_config
	
	
	from models.biencoder import BiEncoderWrapper
	if model_config is not None:
		with open(model_config, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
			
		index, embeds = index_tokenized_entities(biencoder=biencoder,
								 ent_tokens_file=ent_file,
								 force_exact_search=False)
	else:
		embeds = np.load(ent_file)
		index = build_flat_or_ivff_index(embeds=embeds,
										 force_exact_search=False)
		
	temp_search(embeds=embeds, index=index)
	
	# searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()


if __name__ == "__main__":
	main()
