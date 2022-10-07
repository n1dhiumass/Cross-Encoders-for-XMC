# import os
# import sys
# import json
# import torch
# import pickle
# import logging
# import argparse
# import numpy as np
# import networkx as nx
#
# from tqdm import tqdm
# from IPython import embed
# from pathlib import Path
# from networkx.algorithms.shortest_paths.generic import shortest_path
# import matplotlib.pyplot as plt
#
# from models.biencoder import BiEncoderWrapper
# from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
# from eval.eval_utils import compute_label_embeddings
# from eval.nsw_eval_zeshel import get_index, compute_ment_embeds, get_init_ents, compute_ent_embeds_w_tfidf
#
# logging.basicConfig(
# 	stream=sys.stderr,
# 	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
# 	datefmt="%d/%m/%Y %H:%M:%S",
# 	level=logging.INFO,
# )
# LOGGER = logging.getLogger(__name__)
#
# cmap = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
# 		('navy', 'skyblue'), ('olive', 'y'),
# 		('sienna', 'tan'), ('darkviolet', 'orchid'),
# 		('darkorange', 'gold'), ('deeppink', 'violet'),
# 		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]
#
# NUM_ENTS = {"world_of_warcraft" : 27677,
# 			"starwars" : 87056,
# 			"pro_wrestling" : 10133,
# 			"military" : 104520,
# 			"final_fantasy" : 14044,
# 			"fallout" : 16992,
# 			"american_football" : 31929,
# 			"doctor_who" : 40281}
#
#
# def get_graph_stats(nsw_graph, nx_graph):
# 	degrees = [len(v) for v in nsw_graph.values()]
# 	LOGGER.info(f"Degree distribution : \nMean:{np.mean(degrees)} \nPercentile(1):{np.percentile(degrees, 1)} \nPercentile(10):{np.percentile(degrees, 10)} \nPercentile(50):{np.percentile(degrees, 50)} \nPercentile(75):{np.percentile(degrees, 75)} \nPercentile(99):{np.percentile(degrees, 99)}")
# 	degree_info = {"mean": np.mean(degrees),
# 				  "p1":np.percentile(degrees, 1),
# 				  "p10":np.percentile(degrees, 10),
# 				  "p50":np.percentile(degrees, 50),
# 				  "p90":np.percentile(degrees, 90),
# 				  "p99":np.percentile(degrees, 99)}
#
# 	return {"degree": degree_info}
# 	# # n_levels = hnsw_index.index.hnsw.max_level
# 	# # for level in range(n_levels):
# 	# # 	hnsw_index.index.hnsw.print_neighbor_stats(level)
# 	# #
# 	# # LOGGER.info("Computing eccentricity")
# 	# # eccens = distance_measures.eccentricity(nx_graph)
# 	# # LOGGER.info("Computing eccentricity")
# 	# # radius = distance_measures.radius(nx_graph, e=eccens)
# 	# # LOGGER.info(f"Radius : {radius}")
# 	# # diameter = distance_measures.diameter(nx_graph, e=eccens)
# 	# # LOGGER.info(f"Diameter : {diameter}")
# 	# # centers = distance_measures.center(nx_graph, e=eccens)
# 	# # LOGGER.info(f"Center : {center}")
# 	# # sigma_nswness = smallworld.sigma(nx_graph)
# 	# # omega_nswness = smallworld.omega(nx_graph)
# 	#
# 	# # LOGGER.info(f"Sigma_nswness : {sigma_nswness}")
# 	# # LOGGER.info(f"Omega_nswness : {omega_nswness}")
# 	#
# 	# return {"radius":radius,
# 	# 		"diameter":diameter,
# 	# 		"len(centers)":len(centers),
# 	# 		"degree": degree_info}
#
#
# def run_graph_analysis(hnsw_index, nsw_graph, ment_to_ent_scores, gt_labels, ment_embeds, ent_embeds, init_ent_method, n_ents):
# 	"""
# 	Analyze paths from initial entities to ground-truth entities
# 	:param nsw_graph:
# 	:param ment_to_ent_scores:
# 	:param gt_labels:
# 	:param beamsize:
# 	:param init_ents:
# 	:return:
# 	"""
#
# 	try:
# 		nsw_graph = {node:nbrs.tolist() for node, nbrs in nsw_graph.items()}
# 		nx_graph = nx.from_dict_of_lists(nsw_graph)
#
# 		n_ments = len(gt_labels)
#
# 		num_init_vals = [1, 2, 5, 10, 50, 100, 500, 1000]
#
# 		result = {}
# 		LOGGER.info("Finding initial entities")
# 		all_init_ents = get_init_ents(init_ent_method=init_ent_method, ment_embeds=ment_embeds,
# 					  ent_embeds=ent_embeds, k=max(num_init_vals), n_ments=n_ments, n_ents=n_ents)
#
# 		LOGGER.info("Found all initial entities")
# 		all_init_ents = np.array(all_init_ents)
# 		for num_init in num_init_vals:
# 			init_ents = all_init_ents[:,:num_init]
#
# 			gt_in_init_ents = []
# 			all_ment_path_len_stats = []
# 			for ment_idx in range(n_ments):
# 				all_paths = []
# 				for curr_init_ent in init_ents[ment_idx]:
# 					src = curr_init_ent
# 					tgt = gt_labels[ment_idx]
# 					path = shortest_path(G=nx_graph, source=src, target=tgt)
# 					all_paths += [path]
#
# 				# pos_neg_pairs = []
# 				# for path in all_paths:
# 				# 	for node1, node2 in zip(path[:-1], path[1:]):
# 				# 		nbrs_node1 = neighbors(nx_graph, node1)
# 				# 		pos_node  = node2
# 				# 		neg_nodes = [nbr for nbr in nbrs_node1 if nbr != pos_node]
# 				# 		pos_neg_pairs += (pos_node, neg_nodes)
#
# 				path_lengths = [len(path) for path in all_paths]
# 				path_lengths_stats = [np.mean(path_lengths), np.std(path_lengths), np.percentile(path_lengths, 50)]
#
# 				all_ment_path_len_stats += [path_lengths_stats]
#
# 				gt_in_init_ents += [gt_labels[ment_idx] in init_ents[ment_idx]]
#
# 			all_ment_path_len_stats = np.array(all_ment_path_len_stats)
# 			avg_ment_path_len_stats = np.mean(all_ment_path_len_stats, axis=0)
#
# 			gt_in_init_ents = np.array(gt_in_init_ents, dtype=np.int64)
#
# 			result[f"init_ents={num_init}"] = {"gt_in_init_ents": {"mean": np.mean(gt_in_init_ents),
# 																	"p50": np.percentile(gt_in_init_ents, 50),
# 																	"std": np.std(gt_in_init_ents),
# 																	},
# 											  	"path_len_stats" : { "mean":float(avg_ment_path_len_stats[0]),
# 																	 "std":float(avg_ment_path_len_stats[1]),
# 																	 "p50":float(avg_ment_path_len_stats[2])
# 																	 }
# 												}
# 		graph_stats = get_graph_stats(nsw_graph=nsw_graph, nx_graph=nx_graph)
#
# 		final_result = {"graph_stats" : graph_stats}
# 		final_result.update(result)
#
# 		return nx_graph, result
# 	except KeyboardInterrupt:
# 		embed()
# 		return nx_graph, {}
# 	except Exception as e:
# 		embed()
# 		raise e
#
#
#
# def run(embed_type, res_dir, data_info, nsw_metric, biencoder=None):
# 	try:
# 		Path(res_dir).mkdir(exist_ok=True, parents=True)
# 		data_name, data_fname = data_info
#
# 		############################### Read pre-computed cross-encoder score matrix ###################################
# 		LOGGER.info("Loading precomputed ment_to_ent scores")
# 		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
# 			dump_dict = pickle.load(fin)
# 			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
# 			test_data = dump_dict["test_data"]
# 			mention_tokens_list = dump_dict["mention_tokens_list"]
# 			entity_id_list = dump_dict["entity_id_list"]
#
# 		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
# 		# Map entity ids to local ids
# 		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
# 		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
# 		mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]]) for ment_dict in test_data]
#
# 		############################## Compute bi-encoder score matrix #################################################
#
# 		LOGGER.info("Loading precomputed entity encodings computed using biencoder")
# 		# candidate_encoding = np.load(data_fname["ent_embed_file"])
# 		if biencoder:
# 			biencoder.eval()
# 			complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
# 			complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
# 			all_candidate_encoding = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=complete_entity_tokens_list,
# 														  batch_size=200)
#
# 			# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
# 			candidate_encoding = torch.Tensor(all_candidate_encoding[entity_id_list])
# 		else:
# 			all_candidate_encoding = torch.tensor([0.])
# 			candidate_encoding = torch.tensor([0.])
# 		################################################################################################################
# 		result = {}
# 		max_nbrs_vals = [5, 10, 20, 50, None]
#
#
# 		for max_nbrs in tqdm(max_nbrs_vals):
# 			######################################## Build/Read NSW Index on entities ######################################
#
# 			index_path = f"{res_dir}/index_{embed_type}_max_nbrs={max_nbrs}.pkl"
# 			index_path = None # So that we do not save index
# 			index = get_index(
# 				index_path=index_path, embed_type=embed_type,
# 				entity_file=data_fname["ent_file"],
# 				bienc_ent_embeds=all_candidate_encoding.cpu().numpy(),
# 				ment_to_ent_scores=crossenc_ment_to_ent_scores,
# 				max_nbrs=max_nbrs,
# 				graph_metric=nsw_metric
# 			)
#
# 			LOGGER.info("Extracting lowest level NSW graph from index")
# 			# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
# 			nsw_graph = index.get_nsw_graph_at_level(level=1)
# 			################################################################################################################
#
# 			LOGGER.info("Now we will search over the graph")
#
# 			ment_embeds = compute_ment_embeds(embed_type=embed_type, entity_file=data_fname["ent_file"],
# 											  mentions=mentions, biencoder=biencoder, mention_tokens_list=mention_tokens_list)
#
# 			ent_embeds = candidate_encoding.numpy() if embed_type == "bienc" else compute_ent_embeds_w_tfidf(entity_file=data_fname["ent_file"])
#
# 			nx_graph, curr_result = run_graph_analysis(hnsw_index=index, nsw_graph=nsw_graph,
# 													   ment_to_ent_scores=crossenc_ment_to_ent_scores,
# 													   gt_labels=curr_gt_labels, ment_embeds=ment_embeds,
# 													   ent_embeds=ent_embeds, init_ent_method=embed_type, n_ents=n_ents)
#
# 			key = f"max_nbrs={max_nbrs}"
# 			result[key] = {}
# 			result[key].update({"nsw~"+k:v for k,v in curr_result.items()})
#
# 			# LOGGER.info("Saving networkx graph file")
# 			# out_file =  f"{res_dir}/graph_analysis/nx_graph_emb={embed_type}_max_nbrs={max_nbrs}.xml"
# 			# Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
# 			# nx.write_graphml(nx_graph, out_file)
# 			# LOGGER.info("Done saving networkx graph file")
#
# 		out_file = f"{res_dir}/graph_analysis/emb={embed_type}_path_res.json"
# 		out_dir = os.path.dirname(out_file)
# 		Path(out_dir).mkdir(exist_ok=True, parents=True)
#
# 		with open(out_file, "w") as fout:
# 			result["data_info"] = data_info
# 			json.dump(result, fout, indent=4)
# 			LOGGER.info(json.dumps(result,indent=4))
#
#
# 		LOGGER.info("Final embed")
# 		embed()
#
# 	except Exception as e:
# 		embed()
# 		raise e
#
#
#
# def plot(res_dir, data_info):
# 	try:
# 		embed_types = ["tfidf", "bienc", "anchor"]
# 		embed_types = ["tfidf", "bienc"]
# 		# embed_types = ["bienc"]
# 		data_name, data_fnames = data_info
# 		plt_dir = f"{res_dir}/nsw_plots"
# 		# Path(plt_dir).mkdir(exist_ok=True, parents=True)
#
# 		all_results = {}
# 		for embed_type in embed_types:
# 			data_file = f"{res_dir}/graph_analysis/emb={embed_type}.json"
# 			with open(data_file, "r") as fin:
# 				all_results[embed_type] = json.load(fin)
#
# 		_plot_nsw_properties(all_results=all_results, embed_types=embed_types, plt_dir=f"{plt_dir}/nsw_recall_wrt_exact", model_type="crossenc", topk=100)
#
#
# 	except Exception as e:
# 		embed()
# 		raise e
#
#
# def _plot_nsw_properties(all_results, embed_types, plt_dir, model_type, topk):
#
# 	try:
#
# 		beamsize_vals = [1, 2, 5, 10, 50, 100, 500, 1000]
# 		markers = ["o", "*", "s"]
#
# 		max_nbr_vals = [5, 10, 20, 50, None]
# 		budget_vals = [None, 100, 500, 1000, 2000]
#
# 		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
# 		('navy', 'skyblue'), ('olive', 'y'),
# 		('sienna', 'tan'), ('darkviolet', 'orchid'),
# 		('darkorange', 'gold'), ('deeppink', 'violet'),
# 		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]
#
# 		plt.clf()
#
#
# 		for embed_type, c in zip(embed_types, colors):
# 			X, Y = [], []
# 			retr_X, retr_Y = [], []
# 			for max_nbrs in max_nbr_vals:
# 				for beamsize in beamsize_vals:
# 					retr_X += [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"] ]
# 					retr_Y += [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={0}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~acc"]) ]
#
# 					curr_X = [ all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~num_score_comps~mean"]
# 						  for budget in budget_vals]
#
# 					curr_Y = [ float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget={budget}_max_nbrs={max_nbrs}"][f"{model_type}~nsw~acc"])
# 						  for budget in budget_vals]
#
# 					X += curr_X
# 					Y += curr_Y
#
# 					# plt.scatter(curr_X, curr_Y, marker=marker ,c=c)
# 					# plt.scatter(curr_X, curr_Y, c=c, label=f"{embed_type}", alpha=0.6)
#
# 				# plt.scatter(curr_X, curr_Y, marker=marker ,c=c, label=embed_type)
# 				exact_acc = [float(all_results[embed_type][f"k={topk}_b={beamsize}_init_{embed_type}_budget=None_max_nbrs={max_nbrs}"][f"{model_type}~exact~acc"])]*len(X)
# 				plt.plot(X, exact_acc, "-", c=c[1])
#
# 			plt.scatter(X, Y, c=c[1], label=f"nsw", alpha=0.8)
# 			plt.scatter(retr_X, retr_Y, marker="x", c=c[0], alpha=0.8, label=f"init")
#
#
# 		plt.xlabel(f"Number of {model_type} calls")
# 		plt.ylabel(f"Accuracy of {model_type} model using NSW search")
# 		domain = all_results[embed_types[0]]["data_info"][0]
# 		n_ent = int(all_results[embed_types[0]]["data_info"][1][f"crossenc_ment_to_ent_scores"].split("n_e_")[1][:-4])
# 		plt.title(f"Accuracy of {model_type} model using NSW search for domain {domain} w/ {n_ent} entities")
# 		plt.legend()
# 		plt.grid()
# 		out_file = f"{plt_dir}/{model_type}_k={topk}/acc_vs_budget_xslinear.pdf"
# 		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
# 		plt.savefig(out_file)
#
# 		plt.xscale("log")
# 		out_file = f"{plt_dir}/{model_type}_k={topk}/acc_vs_budget_xlog.pdf"
# 		plt.savefig(out_file)
#
#
# 		plt.close()
# 	except Exception as e:
# 		embed()
# 		raise e
#
#
#
# def main():
# 	data_dir = "../../data/zeshel"
#
# 	worlds = get_zeshel_world_info()
#
# 	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
# 	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
# 	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor"], required=True, help="Type of embeddings to use for building NSW")
#
# 	parser.add_argument("--bi_model_config", type=str, default="", help="Biencoder Model config file")
# 	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
# 	parser.add_argument("--nsw_metric", type=str, default="l2", choices=["l2", "ip"], help="Metric/distance to use for building NSW")
#
# 	args = parser.parse_args()
# 	data_name = args.data_name
# 	embed_type = args.embed_type
# 	bi_model_config = args.bi_model_config
# 	nsw_metric = args.nsw_metric
# 	res_dir = args.res_dir
# 	Path(res_dir).mkdir(exist_ok=True, parents=True)
#
# 	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
#
# 	iter_worlds = worlds[:4] if data_name == "all" else [("", data_name)]
#
# 	for world_type, world_name in tqdm(iter_worlds):
# 		LOGGER.info(f"Running inference for world = {world_name}")
# 		if os.path.isfile(bi_model_config):
# 			with open(bi_model_config, "r") as fin:
# 				config = json.load(fin)
# 				biencoder = BiEncoderWrapper.load_model(config=config)
# 		else:
# 			biencoder=None
#
# 		run(res_dir=f"{res_dir}/{world_name}/nsw",
# 			data_info=(world_name, DATASETS[world_name]),
# 			embed_type=embed_type,
# 			biencoder=biencoder,
# 			nsw_metric=nsw_metric)
#
#
# if __name__ == "__main__":
# 	main()
