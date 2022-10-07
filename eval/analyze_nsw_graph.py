import os
import sys
import json
import wandb
import torch
import pickle
import logging
import argparse
import itertools
import numpy as np
import networkx as nx

from tqdm import tqdm
from IPython import embed
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import squareform
from networkx.algorithms.shortest_paths.generic import shortest_path, shortest_path_length
from networkx.algorithms import distance_measures
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from models.biencoder import BiEncoderWrapper
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
from eval.eval_utils import compute_label_embeddings
from eval.nsw_eval_zeshel import get_index, compute_ment_embeds, compute_ent_embeds, get_init_ents, compute_ent_embeds_w_tfidf
from sklearn_extra.cluster import KMedoids


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

cmap = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]

NUM_ENTS = {
	"world_of_warcraft" : 27677,
	"starwars" : 87056,
	"pro_wrestling" : 10133,
	"military" : 104520,
	"final_fantasy" : 14044,
	"fallout" : 16992,
	"american_football" : 31929,
	"doctor_who" : 40281,
	"lego":10076
}


def get_graph_stats(graph, nx_graph):
	degrees = [len(v) for v in graph.values()]
	LOGGER.info(f"Degree distribution : \nMean:{np.mean(degrees)} \n"
				f"Percentile(1):{np.percentile(degrees, 1)} \n"
				f"Percentile(10):{np.percentile(degrees, 10)} \n"
				f"Percentile(50):{np.percentile(degrees, 50)} \n"
				f"Percentile(75):{np.percentile(degrees, 75)} \n"
				f"Percentile(99):{np.percentile(degrees, 99)}")
	degree_info = {"mean": np.mean(degrees),
				  "p1":np.percentile(degrees, 1),
				  "p10":np.percentile(degrees, 10),
				  "p50":np.percentile(degrees, 50),
				  "p90":np.percentile(degrees, 90),
				  "p99":np.percentile(degrees, 99)}
	
	# return {"degree": degree_info}
	# n_levels = hnsw_index.index.hnsw.max_level
	# for level in range(n_levels):
	# 	hnsw_index.index.hnsw.print_neighbor_stats(level)

	LOGGER.info("Computing eccentricity")
	eccens = distance_measures.eccentricity(nx_graph)
	LOGGER.info("Computing eccentricity")
	radius = distance_measures.radius(nx_graph, e=eccens)
	LOGGER.info(f"Radius : {radius}")
	diameter = distance_measures.diameter(nx_graph, e=eccens)
	LOGGER.info(f"Diameter : {diameter}")
	centers = distance_measures.center(nx_graph, e=eccens)
	LOGGER.info(f"Center : {centers}")
	# sigma_nswness = smallworld.sigma(nx_graph)
	# omega_nswness = smallworld.omega(nx_graph)

	# LOGGER.info(f"Sigma_nswness : {sigma_nswness}")
	# LOGGER.info(f"Omega_nswness : {omega_nswness}")

	return {"radius":radius,
			"diameter":diameter,
			"len(centers)":len(centers),
			"centers": list(centers),
			"degree": degree_info}


def get_path_to_gt_ent_stats(nx_graph, all_init_ents, num_init_vals, n_ments, gt_labels):
	"""
	Get stats for path lengths to gt entity from initial seed entities
	:param nx_graph:
	:param all_init_ents:
	:param num_init_vals:
	:param n_ments:
	:param gt_labels:
	:return:
	"""
	result = {}
	all_init_ents = np.array(all_init_ents)
	for num_init in num_init_vals:
		init_ents = all_init_ents[:,:num_init]

		gt_in_init_ents = []
		all_ment_path_len_stats = []
		for ment_idx in range(n_ments):
			all_paths = []
			for curr_init_ent in init_ents[ment_idx]:
				src = curr_init_ent
				tgt = gt_labels[ment_idx]
				path = shortest_path(G=nx_graph, source=src, target=tgt)
				all_paths += [path]
				
			path_lengths = [len(path) for path in all_paths]
			path_lengths_stats = [np.mean(path_lengths), np.std(path_lengths), np.percentile(path_lengths, 50)]
		
			all_ment_path_len_stats += [path_lengths_stats]
			
			gt_in_init_ents += [gt_labels[ment_idx] in init_ents[ment_idx]]
		
		all_ment_path_len_stats = np.array(all_ment_path_len_stats)
		avg_ment_path_len_stats = np.mean(all_ment_path_len_stats, axis=0)
		
		gt_in_init_ents = np.array(gt_in_init_ents, dtype=np.int64)
		
		result[f"init_ents={num_init}"] = {"gt_in_init_ents": {"mean": np.mean(gt_in_init_ents),
																"p50": np.percentile(gt_in_init_ents, 50),
																"std": np.std(gt_in_init_ents),
																},
											"path_len_stats" : { "mean":float(avg_ment_path_len_stats[0]),
																 "std":float(avg_ment_path_len_stats[1]),
																 "p50":float(avg_ment_path_len_stats[2])
																 }
											}
	
	return result


def _get_min_max_stats(list_of_vals):
	return [
		np.min(list_of_vals),
		np.mean(list_of_vals),
		np.std(list_of_vals),
		np.percentile(list_of_vals, 50),
		np.max(list_of_vals)
	]


def _get_path_len_stats(nx_graph, src_nodes, dest_nodes):
	"""
	Return stats on len of paths b/w
		all pairs of src_nodes -OR- all combinations of src_nodes and dest_nodes
	:param nx_graph:
	:param src_nodes:
	:param dest_nodes:
	:return:
	"""
	path_len_stats = {}
	if dest_nodes is None:
		all_paths_lens = [
			shortest_path_length(G=nx_graph, source=src, target=tgt)
			for src, tgt in itertools.combinations(src_nodes, 2)
		]
		all_paths_lens = np.array(all_paths_lens)
		
		
		for mode in ["ap", "mst", "min"]:
			if mode == "ap": # all_pairs
				paths_lens = all_paths_lens
				path_len_stats[mode] = _get_min_max_stats(paths_lens)
			elif mode == "mst" or mode == "min":
				paths_lens = squareform(all_paths_lens)
				np.fill_diagonal(paths_lens, 9999999)
				if mode == "mst":
					tree = minimum_spanning_tree(csr_matrix(paths_lens))
					paths_lens = tree.data
				else:
					paths_lens = np.min(paths_lens, axis=1)
				
				path_len_stats[mode] = _get_min_max_stats(paths_lens)
			else:
				raise NotImplementedError(f"Mode = {mode} not supported")
	else:
		
		all_paths_lens = [
			shortest_path_length(G=nx_graph, source=src, target=tgt)
			for src, tgt in itertools.product(src_nodes, dest_nodes)
		]
		all_paths_lens = np.array(all_paths_lens)
		for mode in ["ap", "mst", "min"]:
			if mode == "ap":
				paths_lens = all_paths_lens
				path_len_stats[mode] = _get_min_max_stats(paths_lens)
			elif mode == "mst" or mode == "min":
				paths_lens = all_paths_lens.reshape(len(src_nodes), len(dest_nodes))
				# We want to find min distance to destination nodes from any start node
				# So taking transpose so that each row contains distane to a dest node from all source nodes
				paths_lens = np.transpose(paths_lens)
				paths_lens = np.min(paths_lens, axis=1)
				path_len_stats[mode] = _get_min_max_stats(paths_lens)
			else:
				raise NotImplementedError(f"Mode = {mode} not supported")
		
		
	# LOGGER.info("Intentional embed")
	# if dest_nodes is not None and not all_pairs: embed()
	# path_len_stats = [
	# 	np.min(paths_lens),
	# 	np.mean(paths_lens),
	# 	np.std(paths_lens),
	# 	np.percentile(paths_lens, 50),
	# 	np.max(paths_lens)
	# ]
	# return path_len_stats
	
	return path_len_stats
	

def get_path_to_top_ent_stats(nx_graph, all_init_ents, num_init_vals, ment_to_ent_scores):
	"""
	Get stats for path lengths to top scoring entity from initial seed entities
	:param nx_graph:
	:param all_init_ents:
	:param num_init_vals:
	:param n_ments:
	:param gt_labels:
	:return:
	"""
	try:
		n_ments, n_ents = ment_to_ent_scores.shape
		seed = 0
		rng = np.random.default_rng(seed)
		result = {}
		all_init_ents = np.array(all_init_ents)
		_, all_top_ents = torch.topk(ment_to_ent_scores, k=max(num_init_vals))
		all_top_ents = all_top_ents.cpu().numpy()
		
		all_rand_ents = rng.integers(low=0, high=n_ents, size=n_ments*max(num_init_vals))
		all_rand_ents = all_rand_ents.reshape(n_ments, max(num_init_vals))
		mode_vals = ["ap", "min", "mst"]
		for num_init in tqdm(num_init_vals):
			init_ents = all_init_ents[:,:num_init]
			top_ents = all_top_ents[:, :num_init] # TODO: Have another variable for this
			rand_ents = all_rand_ents[:, :num_init] # TODO: Add option to sample this multipe times to get better estimates
			
			ent_lists = [("init", init_ents), ("rand", rand_ents), ("top", top_ents)]
			
			len_stats = defaultdict(list)
			for ment_idx in range(n_ments):
				

				# for (ent_type, ent_list), mode in itertools.product(ent_lists, mode_vals):
				for (ent_type, ent_list) in ent_lists:
					# Distance b/w same type of ents
					temp_ans = _get_path_len_stats(nx_graph=nx_graph, src_nodes=ent_list[ment_idx], dest_nodes=None)
					for mode in mode_vals:
						len_stats[f"within_{ent_type}_ents_{mode}"] += [
							temp_ans[mode]
						]
					
				
				for (ent_type1, ent_list1), (ent_type2, ent_list2) in itertools.product(ent_lists, ent_lists):
					if ent_type1 >= ent_type2: continue # Force an ordering b/w ent_type1 and ent_type2
					# Distance b/w type1 and type2 of ents
					temp_ans = _get_path_len_stats(nx_graph=nx_graph, src_nodes=ent_list1[ment_idx], dest_nodes=ent_list2[ment_idx])
					for mode in mode_vals:
						len_stats[f"{ent_type2}_ents_from_{ent_type1}_ents_{mode}"] += [
							temp_ans[mode]
						]
			
			len_stats = {key:( np.mean(np.array(len_stats[key]), axis=0), np.std(np.array(len_stats[key]), axis=0))
						 for key in len_stats}
			
			result[f"init_ents={num_init}_path_len_stats"] = {
				key : {
					"min":(float(len_stats[key][0][0]), float(len_stats[key][1][0])),
					"mean":(float(len_stats[key][0][1]), float(len_stats[key][1][1])),
					"std":(float(len_stats[key][0][2]), float(len_stats[key][1][2])),
					"p50":(float(len_stats[key][0][3]), float(len_stats[key][1][3])),
					"max":(float(len_stats[key][0][4]), float(len_stats[key][1][4])),
				}
				for key in len_stats
			}
		
		return result
	except Exception as e:
		LOGGER.info("Exception in get_path_to_top_ent_stats")
		embed()
		raise e

	
def run_graph_analysis(nx_graph, search_graph, ment_to_ent_scores, gt_labels, ment_embeds, ent_embeds_for_init, init_ent_method, n_ents,
					   num_init_vals):
	"""
	Analyze paths from initial entities to ground-truth entities
	:param search_graph:
	:param ment_to_ent_scores:
	:param gt_labels:
	:param beamsize:
	:param init_ents:
	:return:
	"""
	
	try:
		result = {}
		
		# graph_stats = get_graph_stats(graph=search_graph, nx_graph=nx_graph)
		# result = {"graph_stats": graph_stats}
		# return result
		
		n_ments = len(gt_labels)
		
		LOGGER.info("Finding initial entities")
		all_init_ents = get_init_ents(
			init_ent_method=init_ent_method,
			k=max(num_init_vals),
			ment_embeds=ment_embeds,
			ent_embeds=ent_embeds_for_init,
			n_ments=n_ments,
			n_ents=n_ents,
			force_exact_search=True
		)
		LOGGER.info("Found all initial entities")

		top_ent_path_res = get_path_to_top_ent_stats(
			nx_graph=nx_graph,
			all_init_ents=all_init_ents,
			num_init_vals=num_init_vals,
			ment_to_ent_scores=ment_to_ent_scores
		)
		result.update(top_ent_path_res)
	
		# gt_path_res = get_path_to_gt_ent_stats(
		# 	nx_graph=nx_graph,
		# 	all_init_ents=all_init_ents,
		# 	num_init_vals=num_init_vals,
		# 	n_ments=n_ments,
		# 	gt_labels=gt_labels
		# )
		# result.update(gt_path_res)
		
		return result
	
	except Exception as e:
		embed()
		raise e




def run(embed_type, res_dir, data_info, bi_model_file, graph_metric, graph_type, entry_method, misc, max_nbrs_vals, num_init_vals):
	try:
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fname = data_info

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
				
				
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]]) for ment_dict in test_data]

		wandb.run.summary["status"] = 1

		############################## Compute mention and entity embeddings #################################################
		if bi_model_file != "":
			if bi_model_file.endswith(".json"):
				with open(bi_model_file, "r") as fin:
					biencoder = BiEncoderWrapper.load_model(config=json.load(fin))
			else: # Load from pytorch lightning checkpoint
				biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
			biencoder.eval()
		else:
			biencoder = None
	
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
		else:
			ent_embeds_for_index = None
			
		
		# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
		ent_embeds_for_init = ent_embeds_for_init[entity_id_list] if ent_embeds_for_init is not None else None
		ent_embeds_for_index = ent_embeds_for_index[entity_id_list] if ent_embeds_for_index is not None else None
		
		# ent_embeds = ent_embeds[entity_id_list]
		
		wandb.run.summary["status"] = 2
		################################################################################################################
		result = {}

		for max_nbr_ctr, max_nbrs in tqdm(enumerate(max_nbrs_vals)):
			wandb.log({"max_nbr_ctr": max_nbr_ctr})

			######################################## Build/Read Graph Index on entities ######################################

			LOGGER.info("Building/loading index")
			index_path = None # So that we do not save index
			index = get_index(
				index_path=index_path,
				embed_type=embed_type,
				entity_file=data_fname["ent_file"],
				bienc_ent_embeds=ent_embeds_for_index,
				ment_to_ent_scores=crossenc_ment_to_ent_scores,
  				max_nbrs=max_nbrs,
				graph_metric=graph_metric,
				graph_type=graph_type
			)

			LOGGER.info("Extracting lowest level graph from index")
			# Simulate graph search over this graph with pre-computed cross-encoder scores & Evaluate performance
			search_graph = index.get_nsw_graph_at_level(level=1)
			################################################################################################################

			LOGGER.info("Now we will search over the graph")
			
			search_graph = {node:nbrs.tolist() if  isinstance(nbrs, np.ndarray) else nbrs for node, nbrs in search_graph.items()}
			nx_graph = nx.from_dict_of_lists(search_graph)
			curr_result = run_graph_analysis(
				nx_graph=nx_graph,
				search_graph=search_graph,
				ment_to_ent_scores=crossenc_ment_to_ent_scores,
				gt_labels=curr_gt_labels,
				ment_embeds=ment_embeds,
				ent_embeds_for_init=ent_embeds_for_init,
				init_ent_method=entry_method,
				n_ents=n_ents,
				num_init_vals=num_init_vals
			)

			key = f"max_nbrs={max_nbrs}"
			result[key] = {f"{graph_type}~"+k:v for k,v in curr_result.items()}

			save_graph(
				nx_graph=nx_graph,
				res_dir=res_dir,
				embed_type=embed_type,
				max_nbrs=max_nbrs,
				entity_file=data_fname["ent_file"],
				crossenc_ment_to_ent_scores=crossenc_ment_to_ent_scores,
				ent_embeds_for_index=ent_embeds_for_index,
				misc=misc
			)

		wandb.run.summary["status"] = 3
		out_file = f"{res_dir}/graph_analysis/emb={embed_type}_init={entry_method}_analysis_{max_nbrs_vals}_{num_init_vals}{misc}.json"
		out_file = out_file.replace(" ", "_")
		out_dir = os.path.dirname(out_file)
		Path(out_dir).mkdir(exist_ok=True, parents=True)

		with open(out_file, "w") as fout:
			result["bi_model_file"] = bi_model_file
			result["data_info"] = data_info
			json.dump(result, fout, indent=4)
			LOGGER.info(json.dumps(result,indent=4))

		wandb.run.summary["status"] = 4

	except Exception as e:
		embed()
		raise e



def save_graph(nx_graph, res_dir, embed_type, crossenc_ment_to_ent_scores, entity_file, max_nbrs, ent_embeds_for_index, misc):
	try:
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
		LOGGER.info("Saving networkx graph file")
		out_file =  f"{res_dir}/graph_analysis/nx_graph_emb={embed_type}_max_nbrs={max_nbrs}{misc}.xml"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
	
		
		if embed_type == "tfidf":
			ent_embeds_for_index = compute_ent_embeds_w_tfidf(entity_file=entity_file)
		elif embed_type == "anchor":
			if torch.is_tensor(crossenc_ment_to_ent_scores):
				crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores.cpu().detach().numpy()
			ent_embeds_for_index = np.ascontiguousarray(np.transpose(crossenc_ment_to_ent_scores))
		
		# edge_id_from_attribute = {}
		for node1, node2,weight in nx_graph.edges(data=True):
			nx_graph[node1][node2]['weight'] = np.linalg.norm(ent_embeds_for_index[node1] - ent_embeds_for_index[node2], ord=2)
			# edge_id_from_attribute[node1,node2] = np.linalg.norm(ent_embeds_for_index[node1] - ent_embeds_for_index[node2], ord=2)
		
		if torch.is_tensor(crossenc_ment_to_ent_scores):
				crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores.cpu().detach().numpy()
		anchor_ent_embeds_for_index = np.ascontiguousarray(np.transpose(crossenc_ment_to_ent_scores))
		
		mean_scores =  np.sum(anchor_ent_embeds_for_index, axis=1)
		LOGGER.info(f"Mean_scores {mean_scores}")
		mean_scores  = {idx:score for idx, score in enumerate(mean_scores.tolist())}
		nx.set_node_attributes(nx_graph, mean_scores, "mean_scores")
		
		_, topk_ents = torch.topk( torch.tensor(crossenc_ment_to_ent_scores), k=64)
		topk_ents = topk_ents.numpy().tolist()
		in_topk = {e:-1 for e in range(n_ents)}
		for ment_idx in range(n_ments):
			for e in topk_ents[ment_idx]:
				# in_topk[e] = 1
				# in_topk[e] += 1
				in_topk[e] = ment_idx
				
		# embed()
		nx.set_node_attributes(nx_graph, in_topk, "in_top_k")
		
		# for node in nx_graph.nodes:
		# 	nx_graph[node]['weight'] = np.linalg.norm(anchor_ent_embeds_for_index[node])
		
	
		nx.write_graphml(nx_graph, out_file)
		
		# out_file =  f"{res_dir}/graph_analysis/nx_graph_emb={embed_type}_max_nbrs={max_nbrs}_w_wgts.xml"
		# nx.write_weighted_edgelist(nx_graph, out_file)
		LOGGER.info("Done saving networkx graph file")
	
	except Exception as e:
		embed()
		raise e




def plot(res_dir, data_info, misc, graph_type, entry_method, max_nbrs_vals, num_init_vals):
	try:
		
		# max_nbrs_vals = [5, 10, 20, 50]
		# num_init_vals = [2, 5, 10, 50, 100]
		#
		# max_nbrs_vals = [10]
		# # num_init_vals = [2, 5, 10, 50, 100]
		# num_init_vals = [2, 5], 10, 50, 100]
		
		embed_types = ["tfidf", "bienc" ]
		embed_types = ["anchor", "bienc"]
		data_name, data_fnames = data_info
		plt_dir = f"{res_dir}/graph_analysis/plots"
		Path(plt_dir).mkdir(exist_ok=True, parents=True)
		
		all_results = {}
		for embed_type in embed_types:
			# data_file = f"{res_dir}/graph_analysis/emb={embed_type}_analysis_{max_nbrs_vals}_{num_init_vals}{misc}.json"
			out_file = f"{res_dir}/graph_analysis/emb={embed_type}_init={entry_method}_analysis_{max_nbrs_vals}_{num_init_vals}{misc}.json"
			out_file = out_file.replace(" ", "_")
			try:
				with open(out_file, "r") as fin:
					all_results[embed_type] = json.load(fin)
			except Exception as e:
				LOGGER.info(f"Error for embed_type = {embed_type}")
				
		for max_nbrs_arg in max_nbrs_vals:
			for mode in ["ap", "min", "mst"]:
				_plot_graph_properties(
					all_results=all_results,
					embed_types=embed_types,
					plt_dir=f"{plt_dir}",
					model_type="crossenc",
					topk=100,
					mode=mode,
					num_init_vals=num_init_vals,
					max_nbrs_arg=max_nbrs_arg,
					misc=misc,
					graph_type=graph_type,
					entry_method=entry_method
				)
			
		
	except Exception as e:
		embed()
		raise e
	

def _plot_graph_properties(all_results, embed_types, plt_dir, model_type, topk, mode, max_nbrs_arg, num_init_vals, graph_type, entry_method, misc):
	
	try:
		plt_dir	= f"{plt_dir}/{model_type}_init={entry_method}_k={topk}_max_nbrs_{max_nbrs_arg}{misc}"
		
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]

		
		ent_type_vals = ["init", "top", "rand"]
		# ent_type_vals = ["init", "top"]
		
		# Plotting within entity type stats
		plt.clf()
		width=0.15
		ctr = 0
		max_y = 0
		for embed_type, c in zip(embed_types, colors):
			# for max_nbrs, ent_type in itertools.product(max_nbrs_vals, ent_type_vals):
			for max_nbrs in [max_nbrs_arg]:
				# for ent_type, marker in zip(ent_type_vals, ["x", "s", "o"]):
				for ent_type, hatch in zip(ent_type_vals, ["//", "*", "o"]):
					curr_X = []
					curr_Y = []
					curr_Y_err = []
					for n_init in num_init_vals:
						try:
							temp_X = [n_init]
							temp_y_yerr = all_results[embed_type][f"max_nbrs={max_nbrs}"][f"{graph_type}~init_ents={n_init}_path_len_stats"][f"within_{ent_type}_ents_{mode}"]["mean"]
							temp_Y = [temp_y_yerr[0]]
							temp_Y_err = [temp_y_yerr[1]]
						except Exception as e:
							LOGGER.info(f"Error for embed_type={embed_type}, max_nbrs={max_nbrs}, n_init={n_init} ->{e}")
							# embed()
							# raise e
							temp_X, temp_Y, temp_Y_err = [], [], []
						curr_X += temp_X
						curr_Y += temp_Y
						curr_Y_err += temp_Y_err
					
					curr_X = np.array(curr_X)
					curr_Y = np.array(curr_Y)
					curr_Y_err = np.array(curr_Y_err)
					# plt.plot(curr_X, curr_Y, c=c[0], marker=marker, label=f"{graph_type}~{embed_type}~{ent_type}", alpha=0.6)
					# plt.fill_between(curr_X, curr_Y - curr_Y_err, curr_Y+curr_Y_err, color=c[1], alpha=0.4)
					
					
					X = np.arange(len(curr_Y)) + ctr*width
					plt.bar(X, curr_Y, yerr=curr_Y_err, hatch=hatch, width=width, color=c[1], label=f"{embed_type}~{ent_type}", alpha=0.6, edgecolor=c[0], capsize=4)
					plt.xticks(X, curr_X)
					ctr+=1
					# max_y = max(max_y, max(curr_Y)) if len(curr_Y) > 0 else max_y
		
		# plt.ylim(0, float(np.ceil(max_y))+1)
		# plt.ylim(0, 7)
		plt.xlabel(f"Number of entities")
		plt.ylabel(f"Avg path len b/w entities")
		domain = all_results[embed_types[0]]["data_info"][0]
		n_ent = NUM_ENTS[domain]
		plt.title(f"Avg path len b/w different types of entities in {graph_type} graph for domain {domain} w/ {n_ent} entities")
		plt.legend()
		plt.grid()
		# out_file = f"{plt_dir}/{model_type}_k={topk}_max_nbrs_{max_nbrs_arg}{misc}/within_type_path_len_vs_num_ents_{mode}_{embed_types}_{num_init_vals}.pdf"
		out_file = f"{plt_dir}/within_type_path_len_vs_num_ents_{mode}.pdf"
		out_file = out_file.replace(" ", "_")
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_file)
		
		
		# Plotting cross entity type stats
		plt.clf()
		ctr = 0
		max_y = 0
		for embed_type, c in zip(embed_types, colors):
			X, Y = [], []
			# for max_nbrs, ent_type in itertools.product(max_nbrs_vals, ent_type_vals):
			for max_nbrs in [max_nbrs_arg]:
				# ent_type_pairs = [(ent_type1, ent_type2) for ent_type1, ent_type2 in itertools.permutations(ent_type_vals, 2) if ent_type1 < ent_type2]
				ent_type_pairs = [("init", "top"), ("rand", "top")]
				# for (ent_type1, ent_type2), marker in zip(ent_type_pairs , ["x", "s", "o"]):
				for (ent_type1, ent_type2), hatch in zip(ent_type_pairs , ["//", "*", "o"]):
					curr_X = []
					curr_Y = []
					curr_Y_err = []
					for n_init in num_init_vals:
						try:
							temp_X = [n_init]
							temp_y_yerr = all_results[embed_type][f"max_nbrs={max_nbrs}"][f"{graph_type}~init_ents={n_init}_path_len_stats"][f"{ent_type2}_ents_from_{ent_type1}_ents_{mode}"]["mean"]
							temp_Y = [temp_y_yerr[0]]
							temp_Y_err = [temp_y_yerr[1]]
						except Exception as e:
							LOGGER.info(f"Error for embed_type={embed_type}, max_nbrs={max_nbrs}, n_init={n_init} ->{e}")
							# embed()
							# raise e
							temp_X, temp_Y, temp_Y_err = [], [], []
						curr_X += temp_X
						curr_Y += temp_Y
						curr_Y_err += temp_Y_err
					
					curr_X = np.array(curr_X)
					curr_Y = np.array(curr_Y)
					curr_Y_err = np.array(curr_Y_err)
					# plt.plot(curr_X, curr_Y, c=c[0],marker=marker, label=f"{graph_type}~{embed_type}~{ent_type1}~{ent_type2}", alpha=0.6)
					# plt.fill_between(curr_X, curr_Y - curr_Y_err, curr_Y+curr_Y_err, color=c[1], alpha=0.4)
					
					X = np.arange(len(curr_Y)) + ctr*width
					plt.bar(X, curr_Y, yerr=curr_Y_err, hatch=hatch, width=width, color=c[1], label=f"{embed_type}~{ent_type1}-{ent_type2}", alpha=0.6, edgecolor=c[0], capsize=4)
					plt.xticks(X, curr_X)
					ctr += 1
					# max_y = max(max_y, max(curr_Y)) if len(curr_Y) > 0 else max_y
					
		# plt.ylim(0, float(np.ceil(max_y))+1)
		# plt.ylim(0, 5)
		plt.xlabel(f"Number of entities")
		plt.ylabel(f"Avg path len b/w entities")
		domain = all_results[embed_types[0]]["data_info"][0]
		n_ent = NUM_ENTS[domain]
		plt.title(f"Avg path len b/w different types of entities in {graph_type} graph for domain {domain} w/ {n_ent} entities")
		plt.legend()
		plt.grid()
		out_file = f"{plt_dir}/cross_type_path_len_vs_num_ents_{mode}.pdf"
		out_file = out_file.replace(" ", "_")
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_file)
		
		# plt.xscale("log")
		# out_file = f"{plt_dir}/{model_type}_k={topk}_max_nbrs_{max_nbrs_arg}/cross_type_path_len_vs_num_ents_{mode}_{embed_types}_{num_init_vals}_xlog.pdf"
		# plt.savefig(out_file)
		#
		
		plt.close()
	except Exception as e:
		embed()
		raise e

	
	
def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor"], required=True, help="Type of embeddings to use for building NSW")
	parser.add_argument("--graph_type", type=str, default="nsw", choices=["knn", "nsw", "hnsw", "knn_e2e"], help="Type of graph to use")
	
	parser.add_argument("--bi_model_file", type=str, default="../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt", help="Biencoder Model config file or ckpt file")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions to use for analysis")
	parser.add_argument("--graph_metric", type=str, default="l2", choices=["l2", "ip"], help="Metric/distance to use for building NSW")
	parser.add_argument("--entry_method", type=str, default="bienc", choices=["bienc", "tfidf", "random"], help="Method to choose entry point in graph")
	parser.add_argument("--misc", type=str, default="", help="misc suffix for filename")
	parser.add_argument("--plot_only", type=int, default=0, choices=[0, 1], help="1 to only plot results and 0 to run (h)nsw search and then plot")
	parser.add_argument("--disable_wandb", type=int, default=1, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	args = parser.parse_args()
	data_name = args.data_name
	embed_type = args.embed_type
	bi_model_file = args.bi_model_file
	graph_metric = args.graph_metric
	n_ment = args.n_ment
	res_dir = args.res_dir
	graph_type = args.graph_type
	entry_method = args.entry_method
	disable_wandb = args.disable_wandb
	misc = "_" + args.misc if args.misc != "" else ""
	
	plot_only = bool(args.plot_only)
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	n_ment = n_ment if n_ment != -1 else None
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	
	iter_worlds = worlds[:4] if data_name == "all" else [("", data_name)]
	
	config={
			"goal": "Analyse graph",
			"data_name":data_name,
			"embed_type":embed_type,
			"bi_model_config":bi_model_file,
			"matrix_dir":res_dir,
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(args.__dict__)
	
	wandb.init(
		project="Analyse-Graph",
		dir="../../results/5_CrossEnc/PooledResults",
		config=config,
		mode="disabled" if disable_wandb else "online"
	)
	wandb.run.summary["status"] = 0
	
	# max_nbrs_vals = [5, 10, 20, 50, None]
	max_nbrs_vals = [5, 10, 20, 50]
	num_init_vals = [2, 5, 10, 50, 100]
	
	# max_nbrs_vals = [5, 10]
	# num_init_vals = [2, 5, 10, 50, 100]
	# max_nbrs_vals = [5, 10]
	max_nbrs_vals = [10]
	num_init_vals = [2, 5, 10, 20]
	
	for world_type, world_name in iter_worlds:
		LOGGER.info(f"Running inference for world = {world_name}")
		if not plot_only:
			run(
				res_dir=f"{res_dir}/{world_name}/{graph_type}",
				data_info=(world_name, DATASETS[world_name]),
				embed_type=embed_type,
				bi_model_file=bi_model_file,
				misc=misc,
				graph_metric=graph_metric,
				max_nbrs_vals=max_nbrs_vals,
				num_init_vals=num_init_vals,
				graph_type=graph_type,
				entry_method=entry_method
			)
			
		wandb.run.summary["status"] = 6
		plot(
			res_dir=f"{res_dir}/{world_name}/{graph_type}",
			data_info=(world_name, DATASETS[world_name]),
			max_nbrs_vals=max_nbrs_vals,
			num_init_vals=num_init_vals,
			graph_type=graph_type,
			entry_method=entry_method,
			misc=misc
		)
		wandb.run.summary["status"] = 7


if __name__ == "__main__":
	main()
	# pass
