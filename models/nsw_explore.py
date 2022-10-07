import numpy as np
import faiss

from faiss.contrib import datasets
from IPython import embed

from models.nearest_nbr import HNSWWrapper

def debug():
	# make a 1000-vector dataset in 32D
	# ds = datasets.SyntheticDataset(32, 0, 1000, 0)
	
	d = 10
	n = 10000
	data = np.random.rand(n,d).astype('float32')
	
	
	# Init index and add data
	index = faiss.index_factory(d, "HNSW32") # 32 here signifies float32
	index.add(data)
	
	
	hnsw = index.hnsw
	
	# get nb levels for each vector, and select one
	levels = faiss.vector_to_array(hnsw.levels) # Level for each input
	
	print(levels.max())
	
	print(np.where(levels == 3))
	
	
	
	def get_hnsw_links(hnsw, idx):
		""" get link structure for vertex vno """
		
		# make arrays visible from Python
		levels = vector_to_array(hnsw.levels) # Max level for each indexed vector
		
		# number of neighbors stored per layer (cumulative) - This is a constant for the index and is not specific to any node
		cum_nneighbor_per_level = vector_to_array(hnsw.cum_nneighbor_per_level)
		
		offsets = vector_to_array(hnsw.offsets) # Start and end offset of neighbors of each indexed vector in hnsw.neighbors
		neighbors = vector_to_array(hnsw.neighbors) # List of neighbors of all indexed vectors, across all levels of HNSW
		
		# all neighbors of vector at index idx across all levels
		curr_nbrs = neighbors[offsets[idx]: offsets[idx + 1]]
		
		curr_nbrs_by_levels = [] # List storing list of nbrs of each node in each level of HNSW
		# break down per level
		nlevel = levels[idx]
		for l in range(nlevel):
			# Neighbors of node idx in level l
			curr_level_nbrs  = curr_nbrs[cum_nneighbor_per_level[l] : cum_nneighbor_per_level[l + 1]]
			curr_nbrs_by_levels += [curr_level_nbrs]
		
		return curr_nbrs_by_levels
		
	def vector_to_array(v):
			""" make a vector visible as a numpy array (without copying data)"""
			return faiss.rev_swig_ptr(v.data(), v.size())
	
	nbrs_592 = get_hnsw_links(hnsw, 592)
	embed()


if __name__ == "__main__":
	pass
