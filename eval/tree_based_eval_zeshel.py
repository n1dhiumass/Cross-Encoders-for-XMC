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
from sklearn.feature_extraction.text import TfidfVectorizer



from numpy.linalg import matrix_rank

from eval.eval_utils import score_topk_preds
from utils.data_process import load_entities
from models.nearest_nbr import RandPivotTreeIndex, KMeansPlusPivotTreeIndex

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

cmap = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]


def load_models(biencoder_model, biencoder_config):

	# load biencoder model
	with open(biencoder_config) as json_file:
		biencoder_params = json.load(json_file)
		biencoder_params["path_to_model"] = biencoder_model
	biencoder = load_biencoder(biencoder_params)

	return (
		biencoder,
		biencoder_params,
	)


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
	:param curr_score_tuples: List of (node_id, score) tuple
	:param topk:
	:return:
	"""
	curr_node_n_score_tuples = sorted(curr_node_n_score_tuples, key=lambda x:x[1], reverse=True)
	return curr_node_n_score_tuples[:topk]

	
def compute_ment_embeddings(biencoder, mention_tokens_list):
	with torch.no_grad():
		torch.cuda.empty_cache()
		biencoder.eval()
		bienc_ment_embedding = []
		all_mention_tokens_list_gpu = torch.tensor(mention_tokens_list).to(biencoder.device)
		for ment in all_mention_tokens_list_gpu:
			ment = ment.unsqueeze(0)
			bienc_ment_embedding += [biencoder.encode_input(ment)]
		
		bienc_ment_embedding = torch.cat(bienc_ment_embedding)
	
	return bienc_ment_embedding
	
	
def compute_bienc_ment_to_ent_matrix(biencoder, mention_tokens_list, candidate_encoding):
	bienc_ment_embedding = compute_ment_embeddings(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
	bienc_all_ment_to_ent_scores = bienc_ment_embedding @ candidate_encoding.T
	return bienc_all_ment_to_ent_scores

	
def search_tree(entity_scores, tree_index, top_k, beam_size):
	try:
		top_k_idxs_and_scores, num_score_comps = tree_index.search(query_embed=[], item_scores=entity_scores, top_k=top_k, beam_size=beam_size)
		
		top_k_idxs, top_k_scores = zip(*top_k_idxs_and_scores)
		top_k_idxs = np.array(top_k_idxs)
		top_k_scores = np.array(top_k_scores)
		
		return top_k_scores, top_k_idxs, num_score_comps
	except Exception as e:
		embed()
		raise e
	

def run_tree_search(gt_labels, tree_index, ment_to_ent_scores, rerank_ment_to_ent_scores, top_k, beam_size):
	"""
	
	:param tree_index: Tree index over entities
	:param ment_to_ent_scores:
	:return:
	"""
	
	try:
		n_ments, n_ents = ment_to_ent_scores.shape
		
		# assert len(tree_index) == n_ents, f"Number of entities in NSW graph = {len(nsw_graph)} does not match that in score matrix = {n_ents}"
		
		exact_topk_preds = []
		exact_topk_reranked_preds = []
		tree_topk_preds = []
		tree_topk_reranked_preds = []
		
		
		tree_num_score_comps = []
		for ment_idx in tqdm(range(n_ments)):
			entity_scores = ment_to_ent_scores[ment_idx]
			# Get top-k indices from using Tree index
			tree_topk_scores , tree_topk_ents, tree_curr_num_score_comps = search_tree(entity_scores=entity_scores,
																					   tree_index=tree_index,
																					   top_k=top_k,
																					   beam_size=beam_size)
			
			# Get top-k indices from exact matrix
			exact_topk_scores, exact_topk_ents = entity_scores.topk(top_k)
			
			# Re-rank top-k indices from tree search
			temp = torch.zeros(entity_scores.shape) - 99999999999999
			temp[tree_topk_ents] = rerank_ment_to_ent_scores[ment_idx][tree_topk_ents]
			tree_topk_reranked_scores, tree_topk_reranked_ents = temp.topk(top_k)

			
			# Re-rank top-k indices from exact matrix
			temp = torch.zeros(entity_scores.shape) - 99999999999999
			temp[exact_topk_ents] = rerank_ment_to_ent_scores[ment_idx][exact_topk_ents]
			exact_topk_reranked_scores, exact_topk_reranked_ents = temp.topk(top_k)
			
		
			tree_topk_preds += [(tree_topk_ents, tree_topk_scores)]
			exact_topk_preds += [(exact_topk_ents, exact_topk_scores)]
			
			tree_topk_reranked_preds += [(tree_topk_reranked_ents, tree_topk_reranked_scores)]
			exact_topk_reranked_preds += [(exact_topk_reranked_ents, exact_topk_reranked_scores)]
			
			tree_num_score_comps += [tree_curr_num_score_comps]
			
		tree_topk_preds = _get_indices_scores(tree_topk_preds)
		exact_topk_preds = _get_indices_scores(exact_topk_preds)
		
		tree_topk_reranked_preds = _get_indices_scores(tree_topk_reranked_preds)
		exact_topk_reranked_preds = _get_indices_scores(exact_topk_reranked_preds)
		
		res = {"tree": score_topk_preds(gt_labels=gt_labels,
									   topk_preds={"indices":tree_topk_preds["indices"],
													  "scores":tree_topk_preds["scores"]}),
			   "exact": score_topk_preds(gt_labels=gt_labels,
										 topk_preds={"indices":exact_topk_preds["indices"],
													 "scores":exact_topk_preds["scores"]}),
			   "tree_reranked": score_topk_preds(gt_labels=gt_labels,
												topk_preds={"indices":tree_topk_reranked_preds["indices"],
															"scores":tree_topk_reranked_preds["scores"]}),
			   "exact_reranked": score_topk_preds(gt_labels=gt_labels,
												  topk_preds={"indices":exact_topk_reranked_preds["indices"],
															  "scores":exact_topk_reranked_preds["scores"]}),
			   }
		new_res = {f"{res_type}~{metric}":res[res_type][metric]
				   for res_type in res
				    for metric in res[res_type]}
		
		new_res["tree~num_score_comps~mean"] = np.mean(tree_num_score_comps)
		new_res["tree~num_score_comps~std"] = np.std(tree_num_score_comps)
		for _centile in [1, 10, 50, 90, 99]:
			new_res[f"tree~num_score_comps~p{_centile}"] = np.percentile(tree_num_score_comps, _centile)
		
		return new_res
	except Exception as e:
		embed()
		raise e


def get_dense_tfidf_embeds(entity_file):
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
	
	
	return np.asarray(label_embeds.todense())
	

def get_index(index_path, index_type, embed_type, entity_file, ent_embed_file, ment_to_ent_scores, max_samples_per_leaf):
	"""
	Loads index from given path if available, else builds tree index using entity info and embed_type param
	:param index_path:
	:param embed_type:
	:param entity_file:
	:param ent_embed_file:
	:return:
	"""
	
	try:
		if os.path.isfile(index_path) and False:
			with open(index_path, "rb") as fin:
				index = pickle.load(fin)
		else:
			if embed_type == "tfidf":
				label_embeds = get_dense_tfidf_embeds(entity_file=entity_file)
			elif embed_type == "bienc":
				label_embeds = np.load(ent_embed_file)
			elif embed_type == "anchor":
				if torch.is_tensor(ment_to_ent_scores):
					ment_to_ent_scores = ment_to_ent_scores.cpu().detach().numpy()
				label_embeds = np.ascontiguousarray(np.transpose(ment_to_ent_scores))
			else:
				raise Exception(f"embed_type = {embed_type} not supported")
			
			# Build tree index over it
			LOGGER.info(f"Building Tree Index using entity reps : {label_embeds.shape}")
			if index_type == "rand":
				index = RandPivotTreeIndex(embeds=label_embeds, max_samples_per_leaf=max_samples_per_leaf)
			elif index_type == "kcenter":
				index = KMeansPlusPivotTreeIndex(embeds=label_embeds, max_samples_per_leaf=max_samples_per_leaf)
			else:
				raise Exception(f"Index type = {index_type} not supported.")
			
			LOGGER.info("Now we will save the index")
			with open(index_path, "wb") as fout:
				pickle.dump(index, fout)
		
			LOGGER.info("Finished serializing the object")
	
		return index
	except Exception as e:
		embed()
		raise e


def run(embed_type, index_type, res_dir, data_info, max_samples_per_leaf, biencoder=None):
	try:

		res_dir = f"{res_dir}/emb={embed_type}_pivot={index_type}_leaf_size={max_samples_per_leaf}"
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fnames = data_info
		
		
		############################### Read pre-computed cross-encoder score matrix ###################################
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fnames["crossenc_ment_to_ent_scores"], "rb") as fin:
			(crossenc_ment_to_ent_scores,
			 test_data,
			 mention_tokens_list,
			 entity_id_list,
			 entity_tokens_list) = pickle.load(fin)
		
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		
		
		############################## Compute bi-encoder score matrix #################################################
		
		LOGGER.info("Loading precomputed entity encodings computed using biencoder")
		candidate_encoding = np.load(data_fnames["ent_embed_file"])
		
		# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
		candidate_encoding = torch.Tensor(candidate_encoding[entity_id_list])
		
		bienc_ment_to_ent_scores = compute_bienc_ment_to_ent_matrix(biencoder, mention_tokens_list, candidate_encoding)
	
		################################################################################################################
		
		
		######################################## Build/Read Tree Index on entities ######################################
		
		index_path = f"{res_dir}/index.pkl"
		tree_index = get_index(index_path=index_path, embed_type=embed_type,
							   index_type=index_type,
						  		entity_file=data_fnames["ent_file"],
						  		ent_embed_file=data_fnames["ent_embed_file"],
						  		ment_to_ent_scores=crossenc_ment_to_ent_scores,
								max_samples_per_leaf=max_samples_per_leaf)
		
		################################################################################################################
		
		LOGGER.info("Now we will search over the tree")
		result = {}
		topk_vals = [100]
		beamsize_vals = [1, 2, 5, 10, 20, 50, 100, 200]
		# topk_vals = [100]
		# beamsize_vals = [50]
		for top_k, beam_size in tqdm(itertools.product(topk_vals, beamsize_vals),  total=len(topk_vals)*len(beamsize_vals)):
			
			crossenc_result = run_tree_search(tree_index=tree_index, ment_to_ent_scores=crossenc_ment_to_ent_scores,
											  rerank_ment_to_ent_scores=crossenc_ment_to_ent_scores,
											  gt_labels=curr_gt_labels, top_k=top_k, beam_size=beam_size)
			
			
			bienc_result = run_tree_search(tree_index=tree_index, ment_to_ent_scores=bienc_ment_to_ent_scores,
										   rerank_ment_to_ent_scores=crossenc_ment_to_ent_scores,
										   gt_labels=curr_gt_labels, top_k=top_k, beam_size=beam_size)
			
			
			result[f"k={top_k}_b={beam_size}"] = {}
			result[f"k={top_k}_b={beam_size}"].update({"crossenc~"+k:v for k,v in crossenc_result.items()})
			result[f"k={top_k}_b={beam_size}"].update({"bienc~"+k:v for k,v in bienc_result.items()})
			
			with open(f"{res_dir}/eval.json", "w") as fout:
				result["data_info"] = data_info
				json.dump(result, fout, indent=4)
			
		with open(f"{res_dir}/eval.json", "w") as fout:
			result["data_info"] = data_info
			
			json.dump(result, fout, indent=4)
			LOGGER.info(json.dumps(result,indent=4))
	except Exception as e:
		embed()
		raise e


		
def main():
	exp_id = "4_DomainTransfer"
	pretrained_dir = "../../BLINK_models"
	data_dir = "../../data/zeshel"
	res_dir = f"../../results/{exp_id}"
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	train_worlds =  ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling",
					 "starwars", "world_of_warcraft"]
	test_worlds = ["forgotten_realms", "lego", "star_trek", "yugioh"]
	valid_worlds = ["coronation_street", "elder_scrolls", "ice_hockey", "muppets"]
	
	worlds = [("test",w) for w in test_worlds]
	worlds += [("train",w) for w in train_worlds]
	worlds += [("valid",w) for w in valid_worlds]
	
	DATASETS = {world: {"ment_file": f"{data_dir}/processed/{world_type}_worlds/{world}_mentions.jsonl",
						"ent_file":f"{data_dir}/documents/{world}.json",
						"ent_tokens_file":f"{data_dir}/tokenized_entities/{world}_128_bert_base_uncased.npy",
						"ent_embed_file":f"{data_dir}/tokenized_entities/{world}_128_bert_base_uncased_embeds.npy",
						}
						for world_type, world in worlds
					}
	
	# CrossEncoder score files for some domains/worlds
	DATASETS["lego"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/lego/ment_to_ent_scores_n_m_1000_n_e_10076.pkl"
	DATASETS["star_trek"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/star_trek/ment_to_ent_scores_n_m_400_n_e_34430.pkl"
	DATASETS["forgotten_realms"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/forgotten_realms/ment_to_ent_scores_n_m_100_n_e_15603.pkl"
	DATASETS["yugioh"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/yugioh/ment_to_ent_scores_n_m_1000_n_e_10031.pkl"
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor"], required=True, help="Type of embeddings to use for building index")
	parser.add_argument("--index_type", type=str, choices=["rand", "kcenter"], required=True, help="Type of index tree to build")
	parser.add_argument("--max_samples_per_leaf", type=int, required=True, help="Max items per leaf nodes")
	
	args = parser.parse_args()
	data_name = args.data_name
	embed_type = args.embed_type
	index_type = args.index_type
	max_samples_per_leaf = args.max_samples_per_leaf
	
	biencoder, _ = load_models(biencoder_model=f"{pretrained_dir}/biencoder_wiki_large.bin",
							   biencoder_config=f"{pretrained_dir}/biencoder_wiki_large.json")
	
	if data_name == "all":
		iter_worlds = worlds[:4]
	else:
		iter_worlds = [("", data_name)]
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		run(res_dir=f"{res_dir}/{world_name}/pivot_tree",
			data_info=(world_name, DATASETS[world_name]),
			embed_type=embed_type,
			index_type=index_type,
			biencoder=biencoder,
			max_samples_per_leaf=max_samples_per_leaf)
		

if __name__ == "__main__":
	main()

