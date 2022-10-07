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

NUM_ENTS = {"world_of_warcraft" : 27677,
			"starwars" : 87056,
			"pro_wrestling" : 10133,
			"military" : 104520,
			"final_fantasy" : 14044,
			"fallout" : 16992,
			"american_football" : 31929,
			"doctor_who" : 40281,
			"lego":10076}

	
def get_cluster_indices(embeds, k):
	
	# kmedoids = KMedoids(n_clusters=k, random_state=0, method="pam", init="k-medoids++")
	# kmedoids = KMedoids(n_clusters=k, random_state=0, max_iter=10)
	# kmedoids = KMedoids(n_clusters=k, random_state=0, max_iter=10, method="pam")
	# kmedoids = KMedoids(n_clusters=k, random_state=0, method="pam", init="heuristic")
	kmedoids = KMedoids(n_clusters=k, random_state=0, method="alternate", init="heuristic")
	# kmedoids = KMedoids(n_clusters=k, random_state=0, method="pam", init="heuristic")
	kmedoids = kmedoids.fit(embeds)
	
	cluster_centers = kmedoids.medoid_indices_
	cluster_centers = cluster_centers.tolist() # Convert from numpy array to list
	return cluster_centers, kmedoids
	


def run(embed_type, res_dir, data_info, bi_model_file, num_anchor_vals, misc):
	try:
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fname = data_info

		############################### Read pre-computed cross-encoder score matrix ###################################
		
			
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
	
		
		# 2.)  Compute entity embeddings to use for choosing entry points in graph
		LOGGER.info(f"Computing entity encodings computed using method = {embed_type}")
		if embed_type == "anchor":
			# LOGGER.info("Loading precomputed ment_to_ent scores")
			with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
				dump_dict = pickle.load(fin)
				crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				entity_id_list = dump_dict["entity_id_list"]
				
			if torch.is_tensor(crossenc_ment_to_ent_scores):
				ment_to_ent_scores = crossenc_ment_to_ent_scores.cpu().detach().numpy()
			else:
				ment_to_ent_scores = crossenc_ment_to_ent_scores
			ent_embeds = np.ascontiguousarray(np.transpose(ment_to_ent_scores))
	
			# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
			ent_embeds = ent_embeds[entity_id_list]
		else:
			ent_embeds = compute_ent_embeds(
				embed_type=embed_type,
				biencoder=biencoder,
				entity_tokens_file=data_fname["ent_tokens_file"],
				entity_file=data_fname["ent_file"],
			)
			
		cluster_res = {}
		for ctr, num_anchors in tqdm(enumerate(num_anchor_vals)):
			LOGGER.info(f"Running clustering for num_anchors  = {num_anchors}")
			cluster_res[num_anchors], cluster_params = get_cluster_indices(k=num_anchors, embeds=ent_embeds)
			wandb.run.summary["ctr"] = ctr
			cluster_param_dict = {
				"method": cluster_params.method,
				"init":  cluster_params.init,
				"max_iter":  cluster_params.max_iter,
				"metric":  cluster_params.metric,
				"n_clusters":  cluster_params.n_clusters,
				"random_state":  cluster_params.random_state
			}
			
			cluster_res["cluster_params"] = cluster_param_dict
			with open(f"{res_dir}/anchor_ents_emb={embed_type}_k={num_anchors}_{misc}.json", "w") as fout:
				json.dump(cluster_res, fout)
		
	
	except Exception as e:
		embed()
		raise e

	
def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Compute anchor entities using kmedian clustering')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor"], required=True, help="Type of embeddings to use for building NSW")
	
	parser.add_argument("--bi_model_file", type=str, default="", help="Biencoder Model config file or ckpt file")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions to use for analysis. Usefull only when using anchor embed_type for embedding entities (this requires precomputed ment-ent scores and hence this option requires use to specify n_ment parameter). ")
	parser.add_argument("--misc", type=str, default="", help="misc suffix for filename")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	args = parser.parse_args()
	data_name = args.data_name
	embed_type = args.embed_type
	bi_model_file = args.bi_model_file
	
	n_ment = args.n_ment
	res_dir = args.res_dir
	
	
	disable_wandb = args.disable_wandb
	misc = "_" + args.misc if args.misc != "" else ""
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	n_ment = n_ment if n_ment != -1 else None
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	
	iter_worlds = worlds[:4] if data_name == "all" else [("", data_name)]
	
	num_anchor_vals = [10, 50, 100, 200, 500, 1000]
	num_anchor_vals = [100, 200, 500, 1000]
	
	config={
			"goal": "Compute anchor entities using kmedian clustering",
			"data_name":data_name,
			"embed_type":embed_type,
			"bi_model_config":bi_model_file,
			"num_anchor_vals":num_anchor_vals,
			"matrix_dir":res_dir,
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(args.__dict__)
	
	wandb.init(
		project="Anchor-Entity",
		dir="../../results/5_CrossEnc/PooledResults",
		config=config,
		mode="disabled" if disable_wandb else "online"
	)
	wandb.run.summary["status"] = 0
	
	
	
	for world_type, world_name in iter_worlds:
		LOGGER.info(f"Running inference for world = {world_name}")
		
		run(
			res_dir=f"{res_dir}/{world_name}",
			data_info=(world_name, DATASETS[world_name]),
			embed_type=embed_type,
			bi_model_file=bi_model_file,
			misc=misc,
			num_anchor_vals=num_anchor_vals
		)
		
		wandb.run.summary["status"] = 6
		

if __name__ == "__main__":
	main()
	# pass
