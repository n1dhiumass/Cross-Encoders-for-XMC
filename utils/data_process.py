import os
import csv
import sys
import math
import copy
import time
import json
import torch
import pprint
import logging
import warnings
import argparse
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from IPython import embed
from scipy.special import softmax
from scipy.sparse import load_npz
from pathlib import Path

from networkx.classes.function import neighbors
from networkx.algorithms import distance_measures, smallworld
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from pytorch_transformers.tokenization_bert import BertTokenizer
from xclib.utils.sparse import topk, binarize

from models.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from eval.eval_utils import compute_label_embeddings, compute_input_embeddings
from models.nearest_nbr import build_flat_or_ivff_index, HNSWWrapper
from utils.config import Config

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)



def tokenize_input(input, tokenizer, max_seq_length):
	# 1) Tokenize input and truncate to max_seq_length - 2. We need " - 2" because we need to append [cls] and [sep] token
	input_tokens = tokenizer.tokenize(input)[: max_seq_length - 2]
	
	# 2) Infix input tokens b/w cls and sep token
	input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
	
	# 3) Convert tokens to ids
	input_idxs = tokenizer.convert_tokens_to_ids(input_tokens)
	
	# 4) Add padding if required
	padding = [0] * (max_seq_length - len(input_idxs))  # TODO: Is zero the correct pad?
	input_idxs += padding
	assert len(input_idxs) == max_seq_length
	
	return {"tokens": input_tokens, "token_idxs": input_idxs}


def load_raw_data(config, data_split_type):
	assert isinstance(config, Config)
	if config.data_type == "xmc":
		data_dir = config.data_dir
		if data_split_type == "train":
			input_file = f"{data_dir}/trn.json"
		elif data_split_type == "test":
			input_file = f"{data_dir}/tst.json"
		elif data_split_type == "dev":
			input_file = f"{data_dir}/dev.json"
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		lbl_file = f"{data_dir}/lbl.json"
		print("lower_case", config.lowercase)
		data = read_xmc_data(input_file=input_file, lbl_file=lbl_file, lower_case=config.lowercase)
		return data
	elif config.data_type in ["ent_link"]:
		if data_split_type == "train":
			input_files = config.trn_files
		elif data_split_type == "dev":
			input_files = config.dev_files
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		all_data = {}
		for domain, (mention_file, entity_file, _) in input_files.items():
			mention_data, entity_data = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
			all_data[domain] = (mention_data, entity_data)
			
		return all_data
	elif config.data_type == "ent_link_ce":
		
		if data_split_type == "train":
			domains = config.train_domains
		elif data_split_type == "dev":
			domains = config.dev_domains
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		all_data = {}
		for domain in domains:
			mention_file = config.mention_file_template.format(domain)
			entity_file = config.entity_file_template.format(domain)
			mention_data, entity_data = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
			all_data[f"{data_split_type}~{domain}"] = (mention_data, entity_data)
			
		return all_data
	
	elif config.data_type == "nq":
		if data_split_type == "train":
			input_files = config.trn_files
		elif data_split_type == "dev":
			input_files = config.dev_files
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		assert len(input_files) == 3, f"input_files expected to have 3 entries but has {len(input_files)}"
		
		raw_data_file = input_files[0]
		all_data = {f"nq_{data_split_type}": read_natural_ques_data(filename=raw_data_file)}
		return all_data
	else:
		raise Exception(f"Data type = {config.data_type} is not supported")


def read_xmc_data(input_file, lbl_file, lower_case):
	"""
	Read a extreme multi-label dataset
	:param input_file:
	:param lbl_file:
	:param lower_case: Boolean indicating if we should lowercase data
	:return: Object of class XMCDataset
	"""
	
	labels = []
	with open(lbl_file, "r") as reader:
		for ctr, line in enumerate(reader):
			line_dict = json.loads(line.lower()) if lower_case else json.loads(line.lower())
			labels.append(line_dict["title"])
	
	input_data = []
	with open(input_file, "r") as reader:
		for line in reader:
			line_dict = json.loads(line.lower()) if lower_case else json.loads(line.lower())
			datapoint = {"input": line_dict["title"],
						 "label_idxs": line_dict["target_ind"]
						 }
			input_data.append(datapoint)
	
	return XMCDataset(data=input_data, labels=labels)


def read_ent_link_data(mention_file, entity_file):
	"""
	Load mention and entity data for entity linking
	:param mention_file:
	:param entity_file:
	:return:
	"""
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=entity_file)
	
	mention_data = load_mentions(mention_file=mention_file, kb_id2local_id=kb_id2local_id)
	
	return mention_data, (title2id, id2title, id2text, kb_id2local_id)


def read_natural_ques_data(filename):
	
	with open(filename, "r") as fin:
		data = json.load(fin)
	
	return data


def load_mentions(mention_file, kb_id2local_id):
	"""
	Load mention data
	:param mention_file: Each line contains data about
	:param kb_id2local_id: Dict mapping KB entity id to local entity id. Mention file contains
							 KB id for ground-truth entities so we need to map
							 those KB entity ids to local entity ids
	
	:return: List of mentions.
			Each mention is a dict with four keys :  label_id, context_left, context_right, and mention
	"""
	assert kb_id2local_id and len(kb_id2local_id) > 0, f"kb_id2local_id = {kb_id2local_id} is empty!!!"
	
	test_samples = []
	with open(mention_file, "r") as fin:
		lines = fin.readlines()
		for line in lines:
			record = json.loads(line)
			label_id = record["label_id"]
			
			# check that each entity id (label_id) is in the entity collection
			if label_id not in kb_id2local_id:
				continue
			
			# LOWERCASE EVERYTHING !
			new_record = {"label_id": kb_id2local_id[label_id],
						  "context_left": record["context_left"].lower(),
						  "context_right": record["context_right"].lower(),
						  "mention": record["mention"].lower()
						  }
			test_samples.append(new_record)
	
	LOGGER.info("{}/{} samples considered".format(len(test_samples), len(lines)))
	return test_samples


def load_mentions_by_datasplit(mention_file, all_kb_id2local_id):
	"""
	Load mention data by data split. This loads mentions from various domains in a data split all at once
	:param mention_file: Each line contains data about
	:param all_kb_id2local_id: Dict mapping domain to a
							 dict with KB entity id to local entity id. Mention file contains
							 KB id for ground-truth entities so we need to map
							 those KB entity ids to local entity ids
	
	:return: List of mentions.
			Each mention is a dict with four keys :  label_id, context_left, context_right, and mention
	"""
	assert all_kb_id2local_id and len(all_kb_id2local_id) > 0, f"all_kb_id2local_id = {all_kb_id2local_id} is empty!!!"
	
	test_samples = []
	with open(mention_file, "r") as fin:
		lines = fin.readlines()
		for line in lines:
			record = json.loads(line)
			label_id = record["label_id"]
			domain = record["type"]
			# # check that each entity id (label_id) is in the entity collection
			# if label_id not in kb_id2local_id:
			# 	continue
			
			# LOWERCASE EVERYTHING !
			new_record = {"label_id": all_kb_id2local_id[domain][label_id],
						  "context_left": record["context_left"].lower(),
						  "context_right": record["context_right"].lower(),
						  "mention": record["mention"].lower(),
						  "domain": domain
						  }
			test_samples.append(new_record)
	
	LOGGER.info("{}/{} samples considered".format(len(test_samples), len(lines)))
	return test_samples


def load_entities(entity_file):
	"""
	Load entity data
	:param entity_file: File containing entity data. Each line entity info as a json.
	:return: Dicts
		title2id, id2title, id2text, kb_id2local_id
		kb_id2local_id -> Maps KB entity id to local entity id
	"""
	
	# load all entities in entity_file
	title2id = {}
	id2title = {}
	id2text = {}
	kb_id2local_id = {}
	
	with open(entity_file, "r") as fin:
		lines = fin.readlines()
		for local_idx, line in enumerate(lines):
			entity = json.loads(line)
			
			if "idx" in entity: # For Wikipedia entities
				split = entity["idx"].split("curid=")
				if len(split) > 1:
					kb_id = int(split[-1].strip())
				else:
					kb_id = entity["idx"].strip()
			else: # For ZeShEL entities
				kb_id = entity["document_id"]
			
			
			assert kb_id not in kb_id2local_id
			kb_id2local_id[kb_id] = local_idx
			
			
			title2id[entity["title"]] = local_idx
			id2title[local_idx] = entity["title"]
			id2text[local_idx] = entity["text"]
	
	return (
		title2id,
		id2title,
		id2text,
		kb_id2local_id,
	)


def _get_topk_from_sparse(X, k):
	"""
	Get top-k indices for each row in given scipy sparse matrix
	:param X: Sparse array
	:param k:
	:return:
	"""
	X = X.tocsr()
	X.sort_indices()
	pad_indx = X.shape[1] - 1
	# import pdb
	# pdb.set_trace()
	indices = topk(X, k, pad_indx, 0, return_values=False)
	return indices
	

def get_random_negs(data, n_labels, num_negs, seed, label_key):
	"""
	Sample random negative for each datapoint
	:param data: List of datapoints. Each datapoint is a dictionary storing info such as input text, label_idxs etc
	:param n_labels: Total number of labels
	:param num_negs:
	:param seed:
	:param label_key: Key in datapoint dictionary that stores labels for the corresponding datapoint
	:return:
	"""
	rng = np.random.default_rng(seed)
	
	neg_labels = []
	for datapoint in data:
		p = np.ones(n_labels)
		p[datapoint[label_key]] = 0  # Remove positive labels from list of allowed labels
		p = p / np.sum(p)
		neg_idxs = rng.choice(n_labels, size=num_negs, replace=False, p=p)
		
		# Add neg_labels for this datapoints as many times as the number of positive labels for it
		neg_labels += [neg_idxs]*len(datapoint[label_key]) if isinstance(datapoint[label_key], list) else [neg_idxs]
	
	return neg_labels


def get_random_negs_w_blacklist(n_data, n_labels, num_negs, label_blacklist, seed,):
	"""
	Sample random negative for each datapoint from all labels except for labels blacklisted for corresponding datapoint
	:param n_labels: Total number of labels
	:param num_negs:
	:param seed:
	:param label_blacklist: 2-D List containing labels to ignore for correspondign datapoints
	:return:
	"""
	rng = np.random.default_rng(seed)
	
	neg_labels = []
	for ctr in range(n_data):
		p = np.ones(n_labels)
		p[label_blacklist[ctr]] = 0  # Remove positive labels from list of allowed labels
		p = p / np.sum(p)
		neg_idxs = rng.choice(n_labels, size=num_negs, replace=False, p=p)
		
		neg_labels += [neg_idxs]
	
	return neg_labels


def get_hard_negs_biencoder(biencoder, input_tokens_list, labels_tokens_list, pos_label_idxs, num_negs):
	"""
	Embed inputs and labels, and then mine approx nearest nbr hard negatives for each input
	:param num_negs:
	:param pos_label_idxs:
	:param biencoder
	:param input_tokens_list:
	:param labels_tokens_list:
	:return:
	"""
	
	batch_size = 50

	# Embed tokenized labels and inputs
	label_embeds = compute_label_embeddings(biencoder=biencoder,
											labels_tokens_list=labels_tokens_list,
											batch_size=batch_size)
	
	input_embeds = compute_input_embeddings(biencoder=biencoder,
											input_tokens_list=input_tokens_list,
											batch_size=batch_size)
	
	# Build an index on labels
	nnbr_index = build_flat_or_ivff_index(embeds=label_embeds, force_exact_search=False)
	
	neg_labels = []
	neg_label_scores = []
	for curr_embed, curr_pos_labels in zip(input_embeds, pos_label_idxs):
		curr_pos_labels = set(curr_pos_labels)
		curr_embed = curr_embed.cpu().numpy()[np.newaxis, :]
		
		nn_scores, nn_idxs = nnbr_index.search(curr_embed, num_negs + len(curr_pos_labels))
		
		# Remove positive labels if there are present amongst nearest nbrs
		nn_idx_and_scores = [
							   (nn_idx, nn_score)
							   for nn_idx, nn_score in zip(nn_idxs[0], nn_scores[0])
							   if nn_idx not in curr_pos_labels
						   ][:num_negs]
		nn_idxs, nn_scores = zip(*nn_idx_and_scores)
		nn_idxs, nn_scores = list(nn_idxs), list(nn_scores)
		
		assert len(nn_idxs) == num_negs

		neg_labels += [nn_idxs]
		neg_label_scores += [nn_scores]
	
	neg_labels = np.array(neg_labels)
	neg_label_scores = np.array(neg_label_scores)
	
	return neg_labels, neg_label_scores


def get_hard_negs_tfidf(mentions_data, entity_file, pos_label_idxs, num_negs, force_exact_search=False):
	"""
	Compute hard negatives using tfidf embeddings of entities and mentions
	:return:
	"""
	from eval.nsw_eval_zeshel import compute_ment_embeds, compute_ent_embeds_w_tfidf
	############################# GET MENTION AND ENTITY EMBEDDINGS FOR BUILDING NSW GRAPH #########################
	n_ments = len(mentions_data)
	mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
				for ment_dict in mentions_data]
	
	LOGGER.info(f"Embedding {n_ments} mentions using method = tfidf")
	ment_embeds = compute_ment_embeds(embed_type="tfidf", mentions=mentions, entity_file=entity_file,
									  mention_tokens_list=[], biencoder=None)
	
	LOGGER.info(f"Embedding entities using method = tfidf")
	
	# n_ents = len(ent_embeds)
	################################################################################################################
	
	LOGGER.info(f"Finding {num_negs}+1 nearest entities for {n_ments} mentions from all entities in file {entity_file}")
	nnbr_index = build_flat_or_ivff_index(
		embeds=compute_ent_embeds_w_tfidf(entity_file=entity_file),
		force_exact_search=force_exact_search
	)
	_, init_ents = nnbr_index.search(ment_embeds, num_negs + 1)
	
	# Remove positive labels if there are present amongst nearest nbrs
	final_init_ents = []
	for curr_init_ents, curr_pos_labels in tqdm(zip(init_ents, pos_label_idxs), total=len(pos_label_idxs)):
		curr_init_ents = [ent_idx for ent_idx in curr_init_ents if ent_idx not in curr_pos_labels][:num_negs]
		final_init_ents += [np.array(curr_init_ents)]
	
	final_init_ents = np.array(final_init_ents)
	
	return final_init_ents


def get_nsw_path_train_data(
		embed_type,
		mentions_data,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		max_nbrs,
		biencoder,
		num_paths,
		num_negs_per_node,
		nsw_metric
):
	"""
	Create training data for pairwise model using NSW graph. This version returns pos and negatives for each step
	in the path from seed entity to ground-truth entity
	:param tokenized_entities:
	:param tokenized_mentions:
	:param gt_labels:
	:param max_nbrs:
	:param biencoder:
	:param num_paths:
	:param num_negs_per_node:
	:param mentions_data:
	:param nsw_metric:
	:param embed_type:
	:param entity_file:
	:return:
	"""
	from eval.nsw_eval_zeshel import get_index, compute_ment_embeds, compute_ent_embeds_w_tfidf
	try:
		
		############################# GET MENTION AND ENTITY EMBEDDINGS FOR BUILDING NSW GRAPH #########################
		mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]])
					for ment_dict in mentions_data]
		
		n_ments, n_ents = len(tokenized_mentions), len(tokenized_entities)
		LOGGER.info(f"Embedding {n_ents} entities using method = {embed_type}")
		if embed_type == "bienc":
			ent_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=200)
			ent_embeds = ent_embeds.cpu().detach().numpy()
		elif embed_type == "tfidf":
			ent_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
		else:
			raise NotImplementedError(f"Embed_type = {embed_type} not supported")

		LOGGER.info(f"Embedding {n_ments} mentions using method = {embed_type}")
		ment_embeds = compute_ment_embeds(embed_type=embed_type, mentions=mentions, entity_file=entity_file,
										  mention_tokens_list=tokenized_mentions, biencoder=biencoder)
		
		################################################################################################################
		
		
		############################ BUILD NSW GRAPH AND FINAL INTIAL ENTRY POINTS IN GRAPH ############################
		LOGGER.info(f"Building an NSW index over {n_ents} entities with embed shape {ent_embeds.shape} wiht max_nbrs={max_nbrs}")
		index = get_index(index_path=None, embed_type=embed_type,
						  entity_file=entity_file,
						  bienc_ent_embeds=ent_embeds,
						  ment_to_ent_scores=None,
						  max_nbrs=max_nbrs,
						  graph_metric=nsw_metric)
		
		LOGGER.info("Extracting lowest level NSW graph from index")
		# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
		nsw_graph = index.get_nsw_graph_at_level(level=1)
		nx_graph = nx.from_dict_of_lists(nsw_graph)
		
		LOGGER.info(f"Finding {num_paths} initial entry points in the graph to search for {n_ments} mentions with num_negs_per_node = {num_negs_per_node}")
		# init_ents = get_init_ents(init_ent_method=embed_type, ment_embeds=ment_embeds, ent_embeds=ent_embeds,
		# 						  k=num_paths, n_ments=n_ments, n_ents=n_ents)
		nnbr_index = build_flat_or_ivff_index(embeds=ent_embeds, force_exact_search=False)
		_, init_ents = nnbr_index.search(ment_embeds, num_paths)
		
		################################################################################################################
		
		
		########################## FIND PATHS IN THE GRAPH FROM INITIAL SEED POINTS ####################################
		LOGGER.info("Now finding path from initial entry points to ground-truth entity")
		n_ments = len(tokenized_mentions)
		all_pos_neg_pairs  = []
		all_pos_pair_token_idxs  = []
		all_neg_pair_token_idxs  = []
		total_paths = 0
		for ment_idx in tqdm(range(n_ments), total=n_ments):
			all_paths = []
			# Find path from each seed point to ground-truth entity
			for curr_init_ent in init_ents[ment_idx]:
				src = curr_init_ent
				tgt = gt_labels[ment_idx]
				path = shortest_path(G=nx_graph, source=src, target=tgt)
				all_paths += [path]
			
			# Create list of positive and negatives pairs for each path
			curr_ment_pos_neg_pairs = []
			curr_ment_pos_token_idxs = []
			curr_ment_neg_token_idxs = []
			for path in all_paths:
				if len(path) <= 1: continue
				curr_path_pos_neg_pairs = []
				for node1, node2 in zip(path[:-1], path[1:]):
					nbrs_node1 = neighbors(nx_graph, node1)
					pos_node  = node2
					neg_nodes = [node1] + [nbr for nbr in nbrs_node1 if nbr != pos_node] # Rank node2 higher than node1 and its nbrs other than node2
					while len(neg_nodes) < num_negs_per_node:
						neg_nodes = neg_nodes + neg_nodes
					neg_nodes = neg_nodes[:num_negs_per_node]

					# TODO: Add option for some sub-sampling here so that we can control how many negative we use per positive node in the path
					# TODO: When choosing negative nodes exclude nodes n if there is a path node1-n-node3 as this can be alternative to node1-node2-node3 path and be just as good.
					
					curr_path_pos_neg_pairs += [(pos_node, neg_nodes)]
				
				if len(curr_path_pos_neg_pairs) == 0:
					continue
					
				curr_path_pos_label_idxs, curr_path_neg_labels_idxs = zip(*curr_path_pos_neg_pairs)
				curr_path_pos_label_idxs, curr_path_neg_labels_idxs = np.array(curr_path_pos_label_idxs), list(curr_path_neg_labels_idxs)
				curr_path_pos_paired_token_idxs,  curr_path_neg_paired_token_idxs = _get_paired_token_idxs(tokenized_inputs=tokenized_mentions,
																					   tokenized_labels=tokenized_entities,
																					   pos_label_idxs=curr_path_pos_label_idxs,
																					   neg_labels_idxs=curr_path_neg_labels_idxs)
				
				total_paths += 1
				curr_ment_pos_neg_pairs += [curr_path_pos_neg_pairs]
				curr_ment_pos_token_idxs += curr_path_pos_paired_token_idxs
				curr_ment_neg_token_idxs += curr_path_neg_paired_token_idxs
				
			all_pos_neg_pairs += [curr_ment_pos_neg_pairs]
			all_pos_pair_token_idxs += curr_ment_pos_token_idxs
			all_neg_pair_token_idxs += curr_ment_neg_token_idxs
		
		all_pos_pair_token_idxs  = torch.cat(all_pos_pair_token_idxs)
		all_neg_pair_token_idxs  = torch.cat(all_neg_pair_token_idxs)
		all_pos_neg_pair_token_idxs =  TensorDataset(all_pos_pair_token_idxs, all_neg_pair_token_idxs)
		LOGGER.info("Finished finding pos and neg mention-entity pairs for all mentions")
		return NSWDataset(all_pos_neg_pairs=all_pos_neg_pairs, all_pos_neg_pair_token_idxs=all_pos_neg_pair_token_idxs,
						  total_paths=total_paths, n_ments=n_ments, n_ents=n_ents)
	
	except Exception as e:
		LOGGER.info("Exception in get_nsw_graph_negs")
		# embed()
		raise e


def get_hard_negs_w_dist_in_nsw_graph(
		embed_type,
		mentions_data,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		max_nbrs,
		biencoder,
		num_negs,
		nsw_metric
):
	"""
	Create training data for pairwise model using NSW graph.
	This version returns seed entities along with their distance from ground-truth entity
	:param embed_type:
	:param entity_file:
	:return:
	"""
	from eval.nsw_eval_zeshel import get_index, compute_ent_embeds_w_tfidf
	
	if embed_type == "bienc":
		init_ents, _ = get_hard_negs_biencoder(
			biencoder=biencoder,
			input_tokens_list=tokenized_mentions,
			labels_tokens_list=tokenized_entities,
			pos_label_idxs=[[x] for x in gt_labels],
			num_negs=num_negs
		)
	elif embed_type == "tfidf":
		init_ents = get_hard_negs_tfidf(
			mentions_data=mentions_data,
			entity_file=entity_file,
			pos_label_idxs=[[x] for x in gt_labels],
			num_negs=num_negs
		)
	else:
		raise NotImplementedError(f"Embed type = {embed_type} not supported")
	
		
	############################ BUILD NSW GRAPH AND FINAL INTIAL ENTRY POINTS IN GRAPH ############################
	n_ments, n_ents = len(tokenized_mentions), len(tokenized_entities)
	LOGGER.info(f"Embedding {n_ents} entities using method = {embed_type}")
	if embed_type == "bienc":
		ent_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=200)
		ent_embeds = ent_embeds.cpu().detach().numpy()
	elif embed_type == "tfidf":
		ent_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
	else:
		raise NotImplementedError(f"Embed_type = {embed_type} not supported")
	
	LOGGER.info(f"Building an NSW index over {n_ents} entities with embed shape {ent_embeds.shape} wiht max_nbrs={max_nbrs}")
	index = get_index(index_path=None, embed_type=embed_type,
					  entity_file=entity_file,
					  bienc_ent_embeds=ent_embeds,
					  ment_to_ent_scores=None,
					  max_nbrs=max_nbrs,
					  graph_metric=nsw_metric)
	
	LOGGER.info("Extracting lowest level NSW graph from index")
	# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
	nx_graph = nx.from_dict_of_lists(index.get_nsw_graph_at_level(level=1))
	
	################################################################################################################
	
	
	######################### FIND DISTANCE OF THESE INITIAL SEEDS FROM GT ENTITY ##################################
	all_pos_labels = []
	all_neg_labels = []
	all_neg_label_dists = []
	for ment_idx in tqdm(range(n_ments), total=n_ments):
		gt_node = gt_labels[ment_idx]
		neg_idxs = init_ents[ment_idx]
		neg_idxs_dist = [len(shortest_path(G=nx_graph, source=src, target=gt_node)) for src in neg_idxs]
		
		all_pos_labels += [gt_node]
		all_neg_labels += [neg_idxs]
		all_neg_label_dists += [neg_idxs_dist]
		
		assert all([x>0 for x in neg_idxs_dist])
		
	pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(tokenized_inputs=tokenized_mentions,
																   tokenized_labels=tokenized_entities,
																   pos_label_idxs=all_pos_labels,
																   neg_labels_idxs=all_neg_labels)
	all_neg_label_dists = torch.LongTensor(all_neg_label_dists)
	pos_pair_token_idxs = torch.cat(pos_pair_token_idxs)
	neg_pair_token_idxs = torch.cat(neg_pair_token_idxs)
	dataset =  TensorDataset(pos_pair_token_idxs, neg_pair_token_idxs, all_neg_label_dists)
	LOGGER.info("Finished finding pos and neg mention-entity pairs for all mentions")
	return dataset


def get_hard_negs_w_knn_rank(
		embed_type,
		mentions_data,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		biencoder,
		num_negs

):
	"""
	
	:param embed_type:
	:param mentions_data:
	:param entity_file:
	:param tokenized_mentions:
	:param tokenized_entities:
	:param gt_labels:
	:param biencoder:
	:param num_negs:
	:return:
	"""

	try:
		if embed_type == "bienc":
			neg_ents, _ = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=tokenized_mentions,
				labels_tokens_list=tokenized_entities,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=num_negs
			)
		elif embed_type == "tfidf":
			neg_ents = get_hard_negs_tfidf(
				mentions_data=mentions_data,
				entity_file=entity_file,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=num_negs
			)
		else:
			raise NotImplementedError(f"Embed type = {embed_type} not supported")
	
		pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(
			tokenized_inputs=tokenized_mentions,
			tokenized_labels=tokenized_entities,
			pos_label_idxs=gt_labels,
			neg_labels_idxs=neg_ents
		)
		all_neg_label_dists = torch.LongTensor([ np.arange(1, num_negs+1) for _ in tokenized_mentions])
		pos_pair_token_idxs = torch.cat(pos_pair_token_idxs)
		neg_pair_token_idxs = torch.cat(neg_pair_token_idxs)
		dataset =  TensorDataset(pos_pair_token_idxs, neg_pair_token_idxs, all_neg_label_dists)
		return dataset
	except Exception as e:
		LOGGER.info("Exception in get_hard_negs_w_knn_rank")
		embed()
		raise e


def get_nsw_graph_train_data_w_ranks(
		embed_type,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		max_nbrs,
		biencoder,
		num_negs,
		nsw_metric,
		dist_cutoff,
		seed
):
	"""
	Create training data for pairwise model using NSW graph.
	Training data consist of a list of negatives for each mention with distance of those negatives from gt entity in NSW graph
	:param embed_type: Type of embedding of enitities for building NSW graph
	# :param mentions_data: List of mention data dicts
	:param entity_file: File containing entity information
	:param tokenized_mentions: Tensor with tokenized mentions
	:param tokenized_entities: Tensor with tokenized entities
	:param gt_labels: List of ground-truth entities
	:param max_nbrs: Max nbr parameter while building NSW Graph
	:param biencoder: Biencoder model to use if embed_type == "bienc"
	:param num_negs: Number of negatives to use for each mention
	:param dist_cutoff: Distance cutoff to use when finding shortest path from gt node to other nodes in graph
	:param seed:
	:return:
	"""
	from eval.nsw_eval_zeshel import get_index, compute_ent_embeds_w_tfidf
	try:
		
		############################# GET MENTION AND ENTITY EMBEDDINGS FOR BUILDING NSW GRAPH #########################
		n_ments, n_ents = len(tokenized_mentions), len(tokenized_entities)
		LOGGER.info(f"Embedding {n_ents} entities using method = {embed_type}")
		if embed_type == "bienc":
			ent_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=200)
			ent_embeds = ent_embeds.cpu().detach().numpy()
		elif embed_type == "tfidf":
			ent_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
		else:
			raise NotImplementedError(f"Embed_type = {embed_type} not supported")

		################################################################################################################
		
		
		############################ BUILD NSW GRAPH AND FINAL INTIAL ENTRY POINTS IN GRAPH ############################
		LOGGER.info(f"Building an NSW index over {n_ents} entities with embed shape {ent_embeds.shape} wiht max_nbrs={max_nbrs}")
		index = get_index(
			index_path=None, embed_type=embed_type,
			entity_file=entity_file,
			bienc_ent_embeds=ent_embeds,
			ment_to_ent_scores=None,
			max_nbrs=max_nbrs,
			graph_metric=nsw_metric
		)
		
		LOGGER.info("Extracting lowest level NSW graph from index")
		# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
		nsw_graph = index.get_nsw_graph_at_level(level=1)
		nx_graph = nx.from_dict_of_lists(nsw_graph)
		# avg_degree = np.mean([len(nbrhood) for nbrhood in nsw_graph.values()])
		# LOGGER.info(f"Avg degree of graph = {avg_degree} with max_degree = {max_nbrs}")
		
		################################################################################################################
		
		############################################ FIND NBRS IN THE GRAPH ###########################################
		all_pos_labels = []
		all_neg_labels = []
		all_neg_label_dists = []
		rng = np.random.default_rng(seed)
		for ment_idx in tqdm(range(n_ments), total=n_ments):
			gt_node = gt_labels[ment_idx]
			curr_dist_cutoff = dist_cutoff
			neg_idxs, neg_idxs_dists  = [], []
			while curr_dist_cutoff < n_ments:
				tgt_path_lens = single_source_shortest_path_length(G=nx_graph, source=gt_node, cutoff=curr_dist_cutoff)
				curr_dist_cutoff += 1
				
				if gt_node in tgt_path_lens: tgt_path_lens.pop(gt_node)
				
				tgts_w_dists = list(tgt_path_lens.items())
				if len(tgts_w_dists) < num_negs :
					continue
				
				
				tgts, tgt_dists  = zip(*tgts_w_dists)
				# p = 1/np.array(tgt_dists)
				# # Subsample fixed number of nodes with probability proportional to max_degree**(-1*distance)
				tgt_dists = np.array(tgt_dists)
				p = np.power(1.0*max_nbrs, -tgt_dists)
				p = p/np.sum(p) # Normalize to unit probability
				
				neg_idxs_w_dists = rng.choice(tgts_w_dists, size=num_negs, replace=False, p=p)
				neg_idxs, neg_idxs_dists = zip(*neg_idxs_w_dists)
				
				if len(neg_idxs) >= num_negs:
					break
			
			assert len(neg_idxs) == num_negs, f"neg_idx len = {len(neg_idxs)} but required len = {num_negs}"
			all_pos_labels += [gt_node]
			all_neg_labels += [list(neg_idxs)]
			all_neg_label_dists += [list(neg_idxs_dists)]
			
		pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(tokenized_inputs=tokenized_mentions,
																		   tokenized_labels=tokenized_entities,
																		   pos_label_idxs=all_pos_labels,
																		   neg_labels_idxs=all_neg_labels)
		all_neg_label_dists = torch.LongTensor(all_neg_label_dists)
		pos_pair_token_idxs  = torch.cat(pos_pair_token_idxs)
		neg_pair_token_idxs  = torch.cat(neg_pair_token_idxs)
		dataset =  TensorDataset(pos_pair_token_idxs, neg_pair_token_idxs, all_neg_label_dists)
		LOGGER.info("Finished finding pos and neg mention-entity pairs for all mentions")
		return dataset
	
	except Exception as e:
		LOGGER.info("Exception in get_nsw_graph_negs")
		embed()
		raise e


def get_reranked_hard_negs(
		embed_type,
		mentions_data,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		biencoder,
		num_negs,
		init_num_negs,
		reranker,
		first_segment_end,
		batch_size,
		out_file,
):
	"""
	Mine hard negatives and then re-rank using cross-encoder to retain only top-k negatives wrt crossencoder
	:param embed_type: Entity embedding type to retrieve first set of hard negatives
	:param mentions_data: List of mentions along with their contexts.
	:param entity_file: File containing entity information
	:param tokenized_mentions: List of mention tokens
	:param tokenized_entities: List of entity tokens
	:param gt_labels: List of ground-truth labels for each mention
	:param biencoder: Biencoder model
	:param init_num_negs: Number of negatives to find using initially for purpose of re-ranking
	:param num_negs: Number of negatives to return after re-ranking
	:param reranker: Reranker model
	:param first_segment_end: Input (mention) tokens len in mention-entity paired sequence
	:param batch_size: Batch_size to use when using reranker model to compute scores
	:return:
	"""
	try:
		from models.crossencoder import CrossEncoderWrapper
		
		if embed_type == "bienc":
			neg_ents, _ = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=tokenized_mentions,
				labels_tokens_list=tokenized_entities,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=init_num_negs
			)
		elif embed_type == "tfidf":
			neg_ents = get_hard_negs_tfidf(
				mentions_data=mentions_data,
				entity_file=entity_file,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=init_num_negs
			)
		else:
			raise NotImplementedError(f"Embed type = {embed_type} not supported")
		
		n_ments = len(tokenized_mentions)
		assert neg_ents.shape == (n_ments, init_num_negs), f"neg_ents shape = {neg_ents.shape} does not match (n_ments, init_num_negs) = {n_ments, init_num_negs}"

		######################### RE-RANK THESE NEGATIVES ENTITIES USING CROSS-ENCODER MODEL ###########################
		pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(tokenized_inputs=tokenized_mentions,
																		   tokenized_labels=tokenized_entities,
																		   pos_label_idxs=gt_labels,
																		   neg_labels_idxs=neg_ents)
		
		pos_pair_token_idxs = torch.cat(pos_pair_token_idxs) # Shape: n_ments x seq_len
		neg_pair_token_idxs = torch.cat(neg_pair_token_idxs) # Shape: n_ments x init_num_negs x seq_len
		if reranker:
			if isinstance(reranker, torch.nn.parallel.distributed.DistributedDataParallel):
					reranker = reranker.module.module
			
			assert isinstance(reranker, CrossEncoderWrapper), f"reranker is expected of type CrossEncoderWrapper but is of type = {type(reranker)}"
			torch.cuda.empty_cache()
			LOGGER.info(f"Starting Re-ranking negatives using reranker model scores for {entity_file}")
			dataloader = DataLoader(
				TensorDataset(neg_pair_token_idxs), batch_size=batch_size, shuffle=False
			)
			
			with torch.no_grad():
				all_scores_list = []
				for batch in tqdm(dataloader, position=0, leave=True, total=len(dataloader)):
					batch_input, = batch
					batch_input = batch_input.to(reranker.device)
					
					batch_score = reranker.score_candidate(batch_input, first_segment_end=first_segment_end)
					all_scores_list += [batch_score]
					
				all_scores = torch.cat(all_scores_list)
				assert all_scores.shape == (n_ments, init_num_negs), f"score shape = {all_scores.shape} does not match n_ments, init_num_negs= {n_ments, init_num_negs}"
				
				# Dump this data here
				if out_file is not None:
					try:
						dump = {"scores": all_scores.detach().cpu().numpy().tolist(), "indices":neg_ents.tolist()}
						with open(out_file, "w") as fout:
							json.dump(dump, fout)
						LOGGER.info(f"Successfully saved data in file = {out_file}")
					except Exception as e:
						LOGGER.info(f"Error dumping data into file = {out_file} : {e}")
						
				
				# Find top-k scoring negatives
				topk_scores, topk_score_indices = torch.topk(all_scores, k=num_negs)
				topk_score_indices = topk_score_indices.to(neg_pair_token_idxs.device)
				
				# Select top-k neg pairs for each mention using index_select functionality
				neg_pair_token_idxs = [torch.index_select(neg_pair_token_idxs[i], 0, topk_score_indices[i]).unsqueeze(0)
										   for i in range(n_ments)]
				neg_pair_token_idxs = torch.cat(neg_pair_token_idxs)
			
			LOGGER.info(f"Finished Re-ranking negatives using reranker model scores for {entity_file}")
		else:
			# Keep only num_negs negatives per mention
			neg_pair_token_idxs = neg_pair_token_idxs[:,:num_negs,:]
		
		################################################################################################################
		dataset =  TensorDataset(pos_pair_token_idxs, neg_pair_token_idxs)
		LOGGER.info(f"Finished finding pos and neg mention-entity pairs for all mentions wrt domain = {entity_file}")
		return dataset
	except Exception as e:
		LOGGER.info("Exception in get_reranked_hard_negs")
		embed()
		raise e


def get_negs_w_nsw_search(
		embed_type,
		mentions_data,
		entity_file,
		tokenized_mentions,
		tokenized_entities,
		gt_labels,
		biencoder,
		num_negs,
		reranker,
		first_segment_end,
		comp_budget,
		beamsize,
		max_nbrs,
		nsw_metric,
		n_anchor_ments,
		batch_size=32
):
	"""
	Mine hard negatives wrt using cross-encoder model by searching an NSW graph
	:param embed_type: Entity embedding type to retrieve first set of hard negatives
	:param mentions_data: List of mentions along with their contexts.
	:param entity_file: File containing entity information
	:param tokenized_mentions: List of mention tokens
	:param tokenized_entities: List of entity tokens
	:param gt_labels: List of ground-truth labels for each mention
	:param biencoder: Biencoder model
	:param num_negs: Number of negatives to return after re-ranking
	:param reranker: Reranker model
	:param first_segment_end: Input (mention) tokens len in mention-entity paired sequence
	:param beamsize: Beam size to use for NSW search
	:param max_nbrs: Max nbr parameter to use when building NSW graph
	:param comp_budget: Upper limit on number of reranker calls allowed during NSW search
	:param n_anchor_ments: Number of mentions to use as anchors for computing mention-entity scores when embedding method = anchor
	:param batch_size: Batch size to use during re-ranker score computataions
	:return:
	"""
	
	from eval.nsw_eval_zeshel import compute_ent_embeds_w_tfidf, search_nsw_graph
	from models.crossencoder import CrossEncoderWrapper
	try:
		
		if embed_type == "bienc" or embed_type == "anchor":
			init_ents, _ = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=tokenized_mentions,
				labels_tokens_list=tokenized_entities,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=max(num_negs, beamsize)
			)
		elif embed_type == "tfidf":
			init_ents = get_hard_negs_tfidf(
				mentions_data=mentions_data,
				entity_file=entity_file,
				pos_label_idxs=[[x] for x in gt_labels],
				num_negs=max(num_negs, beamsize)
			)
		else:
			raise NotImplementedError(f"Embed type = {embed_type} not supported")
			
		if reranker:
			if isinstance(reranker, torch.nn.parallel.distributed.DistributedDataParallel):
					reranker = reranker.module.module
			
			assert isinstance(reranker, CrossEncoderWrapper), f"reranker is expected of type CrossEncoderWrapper but is of type = {type(reranker)}"
			
			LOGGER.info(f"Build an NSW graph and searching the graph using scores from re-ranker (crossencoder) model {entity_file}")
			
			######################################### BUILD NSW GRAPH ######################################################
			n_ments, n_ents = len(tokenized_mentions), len(tokenized_entities)
			LOGGER.info(f"Embedding {n_ents} entities using method = {embed_type}")
			if embed_type == "bienc":
				ent_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=200)
				ent_embeds = ent_embeds.cpu().detach().numpy()
			elif embed_type == "tfidf":
				ent_embeds = compute_ent_embeds_w_tfidf(entity_file=entity_file)
			elif embed_type == "anchor":
				# TODO: Re-use this computation while searching NSW for these anchor mentions
				rng = np.random.default_rng(0)
				anchor_ments = rng.choice(n_ments, size=n_anchor_ments, replace=False)
		
				ment_ent_pairs = [create_input_label_pair(input_token_idxs=tokenized_mentions[ment_id],
														  label_token_idxs=tokenized_entities[ent_id]).unsqueeze(0)
								  for ment_id in anchor_ments for ent_id in range(n_ents)]
				
				ment_ent_pairs = torch.cat(ment_ent_pairs).to(reranker.device)
				
				dataloader = DataLoader(TensorDataset(ment_ent_pairs), batch_size=batch_size, shuffle=False)
				ment_to_ent_scores = [reranker.score_candidate(batch_input, first_segment_end=first_segment_end)
									  for (batch_input,) in dataloader]
				ment_to_ent_scores = torch.cat(ment_to_ent_scores)
				ment_to_ent_scores = ment_to_ent_scores.reshape(n_anchor_ments, n_ents)
				if torch.is_tensor(ment_to_ent_scores):
					ment_to_ent_scores = ment_to_ent_scores.cpu().detach().numpy()
				ent_embeds = np.ascontiguousarray(np.transpose(ment_to_ent_scores))
				assert ent_embeds.shape  == (n_ents, n_anchor_ments), \
					f"ent_embeds.shape = {ent_embeds.shape} != (n_ents, n_anchor_ments) = {(n_ents, n_anchor_ments)}"
			else:
				raise NotImplementedError(f"Embed_type = {embed_type} not supported")
			
			LOGGER.info(f"Building an NSW index over {n_ents} entities with embed shape {ent_embeds.shape} with max_nbrs={max_nbrs}")
			dim = ent_embeds.shape[1]
			index = HNSWWrapper(dim=dim, data=ent_embeds, max_nbrs=max_nbrs, metric=nsw_metric)
			
			LOGGER.info("Extracting lowest level NSW graph from index")
			# Simulate NSW search over this graph with pre-computed cross-encoder scores & Evaluate performance
			nsw_graph = index.get_nsw_graph_at_level(level=1)
			
			################################################################################################################
			
			######################### START NSW SEARCH USING CROSS-ENCODER MODEL ###########################################
			
			all_nsw_negs = []
			torch.cuda.empty_cache()
			with torch.no_grad():
				for ment_id in tqdm(range(n_ments), position=0, leave=True, total=n_ments):
					def get_entity_scores(ent_ids):
						
						if len(ent_ids) == 0:
							return []
						all_pairs = []
						for ent_id in ent_ids:
							pair = create_input_label_pair(input_token_idxs=tokenized_mentions[ment_id],
														   label_token_idxs=tokenized_entities[ent_id])
							all_pairs += [pair.unsqueeze(0)]
						all_pairs = torch.cat(all_pairs).to(reranker.device)
						
						# scores = reranker.score_candidate(all_pairs, first_segment_end=first_segment_end).detach().cpu().numpy()
						
						dataloader = DataLoader(TensorDataset(all_pairs), batch_size=batch_size, shuffle=False)
						all_scores_list = [reranker.score_candidate(batch_input, first_segment_end=first_segment_end)
										   for (batch_input,) in dataloader]
						all_scores = torch.cat(all_scores_list)
						assert len(all_scores) == len(ent_ids), f"score shape = {all_scores.shape} does not match len(n_ments) = {len(ent_ids)}"
						
						return all_scores.detach().cpu().numpy()
						
					# Find top num_negs + 1 entities and then filter out gt entity if it is present in top-k
					nsw_topk_scores , nsw_topk_ents, nsw_curr_num_score_comps = search_nsw_graph(
						nsw_graph=nsw_graph,
						entity_scores=get_entity_scores,
						approx_entity_scores_and_masked_nodes=(None,{}),
						topk=num_negs + 1,
						arg_beamsize=beamsize,
						init_ents=np.concatenate(
							(
								init_ents[ment_id],
								np.array([gt_labels[ment_id]])
							)
						),
						comp_budget=comp_budget,
						exit_at_local_minima_arg=False,
						pad_results=True
					)
					nsw_topk_ents = [ent for ent in nsw_topk_ents if ent!=gt_labels[ment_id]] # Remove gt entity from this
					
					assert len(nsw_topk_ents) > 0, f"No entity found in nsw search for ment_id = {ment_id}, {entity_file}"
					while len(nsw_topk_ents) < num_negs:
						nsw_topk_ents += nsw_topk_ents
					
					nsw_topk_ents = nsw_topk_ents[:num_negs]
					assert len(nsw_topk_ents) == num_negs
					all_nsw_negs += [nsw_topk_ents]
				
			pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(
				tokenized_inputs=tokenized_mentions,
				tokenized_labels=tokenized_entities,
				pos_label_idxs=gt_labels,
				neg_labels_idxs=all_nsw_negs
			)
			
			LOGGER.info(f"Finished NSW search for finding NSW negs {entity_file}")
		else:
			LOGGER.info(f"Using entities found using embed_type = {embed_type} as negatives for {entity_file}")
			pos_pair_token_idxs,  neg_pair_token_idxs = _get_paired_token_idxs(
				tokenized_inputs=tokenized_mentions,
				tokenized_labels=tokenized_entities,
				pos_label_idxs=gt_labels,
				neg_labels_idxs=init_ents[:,:num_negs]
			)
			
		################################################################################################################
		
		n_ments = len(tokenized_mentions)
		pos_pair_token_idxs = torch.cat(pos_pair_token_idxs) # Shape: n_ments x seq_len
		neg_pair_token_idxs = torch.cat(neg_pair_token_idxs) # Shape: n_ments x num_negs x seq_len
		
		assert neg_pair_token_idxs.shape[0] == n_ments, f"First dim of neg_pair_token_idxs = {neg_pair_token_idxs.shape[0]} but should be {n_ments}"
		assert neg_pair_token_idxs.shape[1] == num_negs, f"First dim of neg_pair_token_idxs = {neg_pair_token_idxs.shape[1]} but should be {num_negs}"
		
		dataset =  TensorDataset(pos_pair_token_idxs, neg_pair_token_idxs)
		LOGGER.info(f"Finished finding pos and neg mention-entity pairs for all mentions wrt domain = {entity_file}")
		return dataset
	except Exception as e:
		LOGGER.info("Exception in get_negs_w_nsw_search {e}")
		embed()
		raise e



def _sort_by_score(indices, scores):
	"""
	Sort each row in scores array in decreasing order and also permute each row of ent_indices accordingly
	:param indices: 2-D numpy array of indices
	:param scores: 2-D numpy array of scores
	:return:
	"""
	assert indices.shape == scores.shape, f"ent_indices.shape ={indices.shape}  != ent_scores.shape = {scores.shape}"
	n,m = scores.shape
	scores = torch.tensor(scores)
	topk_scores, topk_idxs = torch.topk(scores, m)
	sorted_ent_indices = np.array([indices[i][topk_idxs[i]] for i in range(n)])
	
	return sorted_ent_indices, topk_scores


def get_precomputed_ents_w_scores(ent_w_scores_file, n_ments, tokenized_entities, num_labels):
	"""
	Loads entities and entity scores associated from ent_w_scores_file.
	This can be used for mention and entity biencoder distillation or mention-entity crossencoder training.
	
	:param ent_w_scores_file: File containing entity indices and their scores to use for distillation
	:param n_ments: Number of ments
	:param tokenized_entities: Tensor containing tokenized entities
	:param num_labels: Number of entities to use per mention for purpose of distillation
	:return: ent_indices: shape num_mentions x num_neg_per_mention  array containing entity indices
			entities tensor containing tokenized entities : shape num_mentions x num_neg_per_mention x entity_len
			and entity scores tensor with shape num_mentions x num_neg_per_mention
	"""
	
	with open(ent_w_scores_file, "r") as fin:
		data = json.load(fin)
	
	ent_indices, ent_scores = np.array(data["indices"]), np.array(data["scores"])
	
	ent_indices, ent_scores = _sort_by_score(indices=ent_indices, scores=ent_scores)
	
	ent_indices = ent_indices[:n_ments, :num_labels]
	ent_scores = ent_scores[:n_ments, :num_labels]
	
	assert ent_indices.shape == (n_ments, num_labels), f"ent_indices shape = {ent_indices.shape} " \
													   f"does not match n_ments, num_labels = {n_ments, num_labels}"
	assert ent_indices.shape == ent_scores.shape, f"Indices arrays shape = {ent_indices.shape} is different " \
												  f"from score array shape = {ent_scores.shape}"
	tkn_labels_for_distill  = []
	for ment_idx in range(n_ments):
		# Accumulate tokenizations of neg labels/entities for this mention
		curr_entities = [tokenized_entities[curr_ent_idx].unsqueeze(0)
						 for curr_ent_idx in ent_indices[ment_idx]]
		tkn_labels_for_distill += [torch.cat(curr_entities).unsqueeze(0)]

	tkn_labels_for_distill = torch.cat(tkn_labels_for_distill) # Shape : num_mentions x num_neg_per_mention x entity_len
	
	# Normalize scores in each row to sum up to 1 - Returning unnormalized scores for now. Normalize them if needed when using them later.
	# torch_softmax = torch.nn.Softmax(dim=1)
	# ent_scores  = torch.tensor(ent_scores)
	# ent_scores = torch_softmax(ent_scores)
	# ent_scores = torch.tensor(ent_scores)
	
	return ent_indices, tkn_labels_for_distill, ent_scores


def get_dataloader(config, split_type, raw_data, batch_size, shuffle_data, biencoder, reranker, reranker_batch_size, dump_dir):
	"""
	Create pytorch dataloader object with given raw_data
	:param config:
	:param raw_data: Dict mapping domain identifies to (mention_data, entity_data) tuple
	:param split_type: Data split type
	:param batch_size:
	:param biencoder: Pass None if not using hard negatives
	:param reranker:
	:param reranker_batch_size: Batch_size to use when computing scores using re-ranker model
	:param shuffle_data: Shuffle data in dataloaders
	:param dump_dir: dir to dump some computations
	
	:return: Object of type DataLoader
	"""
	tokenizer = BertTokenizer.from_pretrained(config.bert_model,
											  do_lower_case=config.lowercase)
	
	if config.data_type == "xmc":
		if split_type == "dev":
			precomp_fname = config.dev_precomp_top_labels_fname
		elif split_type == "train":
			precomp_fname = config.train_precomp_top_labels_fname
		else:
			raise NotImplementedError(f"Split type = {split_type} not supported")
		
		dataset = get_xmc_dataset(
			model_type=config.model_type,
			input_data=raw_data.data,
			labels=raw_data.labels,
			tokenizer=tokenizer,
			max_input_len=config.max_input_len,
			max_label_len=config.max_label_len,
			biencoder=biencoder,
			neg_strategy=config.neg_strategy,
			num_negs=config.num_negs,
			pos_strategy=config.pos_strategy,
			max_pos_labels=config.max_pos_labels,
			total_labels_per_input=config.total_labels_per_input,
			precomp_fname=precomp_fname
		)

		dataloader = DataLoader(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=shuffle_data
		)
	
		return dataloader
	
	elif config.data_type == "ent_link":
		all_datasets = []
		for domain in sorted(raw_data):
			(mention_data, entity_data) = raw_data[domain]
			(title2id, id2title, id2text, kb_id2local_id) = entity_data
			
			
			if domain in config.trn_files:
				entity_file = config.trn_files[domain][1]
				ent_tokens_file = config.trn_files[domain][2]
			elif domain in config.dev_files:
				entity_file = config.dev_files[domain][1]
				ent_tokens_file = config.dev_files[domain][2]
			else:
				raise NotImplementedError(f"Domain ={domain} not present in "
										  f"train domains = {list(config.trn_files.keys())} or "
										  f"dev domains = {list(config.trn_files.keys())}")
			
			if dump_dir is not None:
				out_file = f"{dump_dir}/{domain}_dump.json"
				Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			else:
				out_file = None
				
			dataset =  get_ent_link_dataset(
				raw_data=raw_data[domain],
				tokenizer=tokenizer,
				neg_strategy=config.neg_strategy,
				ent_tokens_file=ent_tokens_file,
				model_type=config.model_type,
				max_input_len=config.max_input_len,
				max_label_len=config.max_label_len,
				seed=config.seed,
				num_negs=config.num_negs,
				ent_w_score_file=config.ent_w_score_file_template.format(domain),
				biencoder=biencoder,
				nsw_args={
					"max_nbrs":config.nsw_max_nbrs,
					"entity_file":entity_file,
					"embed_type":config.nsw_embed_type,
					"num_paths":config.nsw_num_paths,
					"num_negs_per_node":config.num_negs_per_node,
					"dist_cutoff":config.dist_cutoff,
					"beamsize":config.nsw_beamsize,
					"comp_budget":config.nsw_comp_budget,
					"n_anchor_ments":config.n_anchor_ments,
					"nsw_metric": config.nsw_metric,
				},
				reranker_args={
					"init_num_negs": config.init_num_negs,
					"reranker": reranker,
					"reranker_batch_size": reranker_batch_size
				},
				distill_args={
					"n_labels": config.distill_n_labels
				},
				out_file=out_file
			)

			all_datasets += [dataset]
		
		if isinstance(all_datasets[0], TensorDataset):
			num_neg_splits = config.num_neg_splits
			if num_neg_splits > 1:
				LOGGER.info(f"Reshaping number of negatives per example from {config.num_negs} to {config.num_negs/config.num_neg_splits}")
				new_split_datasets = []
				for dataset in all_datasets:
					new_dataset = _split_negs_into_multiple_batches(
						dataset=dataset,
						num_neg_splits=num_neg_splits,
						neg_strategy=config.neg_strategy
					)
					
					new_split_datasets += [new_dataset]
				all_datasets = new_split_datasets
			
			all_datasets = ConcatDataset(all_datasets)
			dataloader = DataLoader(dataset=all_datasets,
									shuffle=shuffle_data,
									batch_size=batch_size)
		
			return dataloader
		elif isinstance(all_datasets[0], ConcatDataset):
			num_neg_splits = config.num_neg_splits
			assert num_neg_splits <= 1, f"Splitting of negatives not supported so num_neg_splits = {num_neg_splits} should be <=1"
			all_datasets = ConcatDataset(all_datasets)
			dataloader = DataLoader(
				dataset=all_datasets,
				shuffle=shuffle_data,
				batch_size=batch_size
			)
			return dataloader

		# elif isinstance(all_datasets[0], NSWDataset):
		# 	raise NotImplementedError(f"No longer supported datasets of type={type(all_datasets[0])}")
		# 	# This is probably the case for NSW based training of cross-encoder models where
		# 	# we are not creating a tensor due to irregular number of negatives
		# 	# return NSWDataset.concat(all_datasets=all_datasets)
		# 	all_datasets = NSWDataset.concat(all_datasets=all_datasets)
		# 	return DataLoader(dataset=all_datasets.all_pos_neg_pair_token_idxs,
		# 					  shuffle=shuffle_data,
		# 					  batch_size=batch_size)
		else:
			raise Exception(f"Invalid type of dataset = {type(all_datasets[0])}")
	
	elif config.data_type == "ent_link_ce":
		all_datasets = []
		# For training on entity linking datasets on cross-encoder models
		for split_n_domain in sorted(raw_data):
			# (mention_data, entity_data) = raw_data[domain]
			# (title2id, id2title, id2text, kb_id2local_id) = entity_data
			split, domain =  split_n_domain.split("~")
			
			ent_tokens_file = config.entity_token_file_template.format(domain)
			if split == "train":
				ent_w_score_file = config.train_ent_w_score_file_template.format(domain)
			elif split == "dev":
				ent_w_score_file = config.dev_ent_w_score_file_template.format(domain)
			else:
				raise NotImplementedError(f"split={split} not supported")
			
			
			dataset =  get_ent_link_ce_dataset(
				raw_data=raw_data[split_n_domain],
				tokenizer=tokenizer,
				neg_strategy=config.neg_strategy,
				ent_tokens_file=ent_tokens_file,
				model_type=config.model_type,
				max_input_len=config.max_input_len,
				max_label_len=config.max_label_len,
				seed=config.seed,
				num_negs=config.num_negs,
				ent_w_score_file=ent_w_score_file,
				biencoder=biencoder,
				num_pos_labels_for_distill=config.distill_n_labels,
			)

			all_datasets += [dataset]
		
		
		assert config.num_neg_splits <= 1, f"Splitting of negatives not supported so num_neg_splits = {config.num_neg_splits} should be <=1"
		if isinstance(all_datasets[0], TensorDataset) or isinstance(all_datasets[0], ConcatDataset):
			all_datasets = ConcatDataset(all_datasets)
			dataloader = DataLoader(
				dataset=all_datasets,
				shuffle=shuffle_data,
				batch_size=batch_size
			)
			return dataloader
		else:
			raise Exception(f"Invalid type of dataset = {type(all_datasets[0])}")
	
	elif config.data_type == "nq":
		
		assert len(raw_data) == 1, f"raw_data expected to have only 1 key = nq but found {len(raw_data)} keys"
		if "nq_train" in raw_data:
			raw_data = raw_data["nq_train"]
			tknzd_q_file = config.trn_files[1]
			tknzd_psg_file = config.trn_files[2]
		elif "nq_dev" in raw_data:
			raw_data = raw_data["nq_dev"]
			tknzd_q_file = config.dev_files[1]
			tknzd_psg_file = config.dev_files[2]
		else:
			raise NotImplementedError(f"raw_data key ={raw_data.keys()} not supported")
		
		dataset = get_passage_retrieval_dataset(
			tokenizer=tokenizer,
			raw_data=raw_data,
			model_type=config.model_type,
			tknzd_q_file=tknzd_q_file,
			tknzd_psg_file=tknzd_psg_file,
			num_negs=config.num_negs,
			max_input_len=config.max_input_len,
			max_label_len=config.max_label_len,
			use_top_negs=config.use_top_negs,
			seed=config.seed
		)
		dataloader = DataLoader(
			dataset=dataset,
			shuffle=shuffle_data,
			batch_size=batch_size
		)
		return dataloader
	else:
		raise Exception(f"Data type = {config.data_type} is not supported")


def _split_negs_into_multiple_batches(dataset, num_neg_splits, neg_strategy):
	"""
	Split negatives across multiple batches.
	For eg if num_negs = 64 and num_neg_splits=4, then 4 new batches each with 64/4 = 16 negatives is created.
	:param dataset: Tendor dataset
	:param num_neg_splits: Number of mini-batches to split each batch in
	:param neg_strategy: This describes how negatives were mined when creating dataset
	:return:
	"""
	if neg_strategy in ["bienc_distill"]:
		pair_idxs, pair_scores = dataset.tensors
		
		# neg shape = num_examples, num_negs_per_example, seq_len
		num_pairs = pair_idxs.shape[1]
		assert num_pairs % num_neg_splits == 0, f"Num input-label pairs per example = {num_pairs} is not" \
												   f" divisible by num_neg_splits={num_neg_splits}"
		split_size = int(num_pairs/num_neg_splits)
		pair_idxs_split = torch.split(pair_idxs, split_size, dim=1) # Split along num_negs_per_example dim which is dim 1
		pair_scores_split = torch.split(pair_scores, split_size, dim=1) # Split along num_negs_per_example dim which is dim 1
		
		final_pair_idxs = torch.cat(pair_idxs_split, dim=0)
		final_pair_scores = torch.cat(pair_scores_split, dim=0)
	
		new_dataset = TensorDataset(final_pair_idxs, final_pair_scores)
		return new_dataset
	else:
		pos, neg = dataset.tensors
		
		# neg shape = num_examples, num_negs_per_example, seq_len
		num_negs_per_eg = neg.shape[1]
		assert num_negs_per_eg % num_neg_splits == 0, f"Num negs per example = {num_negs_per_eg} is not" \
												   f" divisible by num_neg_splits={num_neg_splits}"
		split_size = int(num_negs_per_eg/num_neg_splits)
		neg_split = torch.split(neg, split_size, dim=1) # Split along num_negs_per_example dim which is dim 1
		
		final_negs = torch.cat(neg_split, dim=0)
		final_pos = torch.cat([pos]*num_neg_splits, dim=0)
		
		new_dataset = TensorDataset(final_pos, final_negs)
		return new_dataset
	
	
	
	return new_dataset


def get_ent_link_dataset(
	model_type,
	tokenizer,
	raw_data,
	ent_tokens_file,
	biencoder,
	neg_strategy,
	num_negs,
	max_input_len,
	max_label_len,
	ent_w_score_file,
	seed,
	nsw_args,
	reranker_args,
	distill_args,
	out_file
):
	"""
	Get dataset with tokenized data for entity linking using raw data.
	It first tokenizes the dataset and then creates a dataset with positive/negative training datapoints
	:param raw_data:
	:param tokenizer
	:param ent_tokens_file:
	:param model_type:
	:param max_input_len
	:param max_label_len:
	:param neg_strategy:
	:param num_negs:
	:param biencoder:
	:param reranker_args:
	:param nsw_args:
	:param distill_args:
	:param seed:
	:param ent_w_score_file
	:return: Object of type TensorDataset
	"""
	try:
		mention_data, (title2id, id2title, id2text, kb_id2local_id) = raw_data
		
		#################################### TOKENIZE MENTIONS AND ENTITIES ############################################
		
		LOGGER.info("Loading and tokenizing mentions")
		tokenized_mentions = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=max_input_len,)["ids"]
								for mention in tqdm(mention_data)]
		LOGGER.info("Finished tokenizing mentions")
		tokenized_mentions = torch.LongTensor(tokenized_mentions)
		
		if ent_tokens_file is not None and os.path.isfile(ent_tokens_file):
			LOGGER.info(f"Reading tokenized entities from file {ent_tokens_file}")
			tokenized_entities = torch.LongTensor(np.load(ent_tokens_file))
		else:
			LOGGER.info(f"Tokenizing {len(id2title)} entities")
			tokenized_entities = [ get_candidate_representation(candidate_title=id2title[ent_id],
																candidate_desc=id2text[ent_id],
																tokenizer=tokenizer,
																max_seq_length=max_label_len)["ids"]
									for ent_id in tqdm(sorted(id2title))]
			tokenized_entities = torch.LongTensor(tokenized_entities)
		
		################################################################################################################
		
		######################### GENERATE POSITIVE AND NEGATIVE LABELS FOR EACH DATAPOINT #############################
		
		

		if neg_strategy == "nsw_graph_path":
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			max_nbrs = nsw_args["max_nbrs"]
			num_paths = nsw_args["num_paths"]
			embed_type = nsw_args["embed_type"]
			entity_file = nsw_args["entity_file"]
			num_negs_per_node = nsw_args["num_negs_per_node"]
			nsw_metric = nsw_args["nsw_metric"]
			
			nsw_dataset = get_nsw_path_train_data(
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				gt_labels=pos_label_idxs,
				embed_type=embed_type,
				entity_file=entity_file,
				max_nbrs=max_nbrs,
				nsw_metric=nsw_metric,
				biencoder=biencoder,
				num_paths=num_paths,
				mentions_data=mention_data,
				num_negs_per_node=num_negs_per_node,
			)
			return nsw_dataset
		elif neg_strategy == "nsw_graph_rank":
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			max_nbrs = nsw_args["max_nbrs"]
			embed_type = nsw_args["embed_type"]
			entity_file = nsw_args["entity_file"]
			dist_cutoff = nsw_args["dist_cutoff"]
			nsw_metric = nsw_args["nsw_metric"]
			
			nsw_dataset = get_nsw_graph_train_data_w_ranks(
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				gt_labels=pos_label_idxs,
				embed_type=embed_type,
				entity_file=entity_file,
				max_nbrs=max_nbrs,
				nsw_metric=nsw_metric,
				biencoder=biencoder,
				num_negs=num_negs,
				dist_cutoff=dist_cutoff,
				seed=seed
			)
			return nsw_dataset
		elif neg_strategy == "hard_negs_w_nsw_rank":
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			max_nbrs = nsw_args["max_nbrs"]
			embed_type = nsw_args["embed_type"]
			entity_file = nsw_args["entity_file"]
			nsw_metric = nsw_args["nsw_metric"]
			
			dataset = get_hard_negs_w_dist_in_nsw_graph(
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				mentions_data=mention_data,
				gt_labels=pos_label_idxs,
				embed_type=embed_type,
				nsw_metric=nsw_metric,
				entity_file=entity_file,
				max_nbrs=max_nbrs,
				biencoder=biencoder,
				num_negs=num_negs
			)
			return dataset
		elif neg_strategy in ["bienc_hard_negs_w_rerank", "tfidf_hard_negs_w_rerank"]:
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			embed_type = "bienc" if neg_strategy == "bienc_hard_negs_w_rerank" else "tfidf"
			entity_file = nsw_args["entity_file"]
			init_num_negs = reranker_args["init_num_negs"]
			reranker = reranker_args["reranker"]
			batch_size = reranker_args["reranker_batch_size"]
			out_file = out_file
			dataset = get_reranked_hard_negs(
				embed_type=embed_type,
				mentions_data=mention_data,
				entity_file=entity_file,
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				gt_labels=pos_label_idxs,
				biencoder=biencoder,
				reranker=reranker,
				first_segment_end=max_input_len,
				num_negs=num_negs,
				init_num_negs=init_num_negs,
				batch_size=batch_size,
				out_file=out_file
			)
			return dataset
		elif neg_strategy in ["bienc_hard_negs_w_knn_rank", "tfidf_hard_negs_w_knn_rank"]:
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			embed_type = neg_strategy[:5]
			entity_file = nsw_args["entity_file"]
			
			dataset = get_hard_negs_w_knn_rank(
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				mentions_data=mention_data,
				gt_labels=pos_label_idxs,
				embed_type=embed_type,
				entity_file=entity_file,
				biencoder=biencoder,
				num_negs=num_negs
			)
			return dataset
		elif neg_strategy in ["bienc_nsw_search", "tfidf_nsw_search", "anchor_nsw_search"]:
			pos_label_idxs = [int(mention["label_id"]) for mention in mention_data]
			embed_type = neg_strategy[:-11]
			assert embed_type in ["bienc", "tfidf", "anchor"], f"embed_type = {embed_type} not supported"
			
			reranker = reranker_args["reranker"]
			entity_file = nsw_args["entity_file"]
			beamsize = nsw_args["beamsize"]
			comp_budget = nsw_args["comp_budget"]
			max_nbrs = nsw_args["max_nbrs"]
			n_anchor_ments = nsw_args["n_anchor_ments"]
			nsw_metric = nsw_args["nsw_metric"]
			
			dataset = get_negs_w_nsw_search(
				embed_type=embed_type,
				mentions_data=mention_data,
				entity_file=entity_file,
				tokenized_mentions=tokenized_mentions,
				tokenized_entities=tokenized_entities,
				gt_labels=pos_label_idxs,
				biencoder=biencoder,
				reranker=reranker,
				first_segment_end=max_input_len,
				num_negs=num_negs,
				beamsize=beamsize,
				comp_budget=comp_budget,
				max_nbrs=max_nbrs,
				nsw_metric=nsw_metric,
				n_anchor_ments=n_anchor_ments
			)
			return dataset
		elif neg_strategy in ["distill", "ent_distill"]:
			LOGGER.info("Loading data for knowledge distillation")
			assert model_type == "bi_enc", f"Model_type = {model_type} is not  supported for neg_strategy = {neg_strategy}"
			num_labels_for_distill = distill_args["n_labels"]
			
			ent_indices, tkn_labels_for_distill, ent_scores = get_precomputed_ents_w_scores(
				ent_w_scores_file=ent_w_score_file,
				num_labels=num_labels_for_distill,
				n_ments=len(tokenized_mentions),
				tokenized_entities=tokenized_entities
			)
			dataset = TensorDataset(tokenized_mentions, tkn_labels_for_distill, ent_scores)
			return dataset
		elif neg_strategy in ["top_ce_as_pos_w_bienc_hard_negs"]:
			LOGGER.info("Loading data for knowledge distillation by treating top-k crossenc entities as positive and otherwise doing hard neg mining for biencoder")
			assert model_type == "bi_enc", f"Model_type = {model_type} is not  supported for neg_strategy = {neg_strategy}"
			num_labels_for_distill = distill_args["n_labels"]
			
			top_ce_ent_indices, tkn_labels_for_distill, top_ce_ent_scores = get_precomputed_ents_w_scores(
				ent_w_scores_file=ent_w_score_file,
				num_labels=num_labels_for_distill,
				n_ments=len(tokenized_mentions),
				tokenized_entities=tokenized_entities
			)
			
			if biencoder is None:
				warnings.warn("Mining negative randomly as biencoder model is not provided")
				raise NotImplementedError("Should use def get_random_negs_w_blacklist(n_data, n_labels, num_negs, label_blacklist, seed,) function here")
				total_n_labels = len(tokenized_entities)
				neg_labels_idxs = get_random_negs(
					data=mention_data,
					seed=0,
					num_negs=num_negs,
					n_labels=total_n_labels,
					label_key="label_id"
				)
			else:
				# Get hard negatives for biencoder while treating top-cross-encoder labels as positive labels
				neg_labels_idxs, _ = get_hard_negs_biencoder(
					biencoder=biencoder,
					input_tokens_list=tokenized_mentions,
					labels_tokens_list=tokenized_entities,
					pos_label_idxs=top_ce_ent_indices,
					num_negs=num_negs
				)
			
			# top_ce_ent_indices.shape == n_data, num_labels_for_distill
			# neg_labels_idxs.shape == n_data, num_labels_for_distill
		
			all_datasets = []
			for pos_ctr_iter in range(num_labels_for_distill):
				curr_pos_label_idxs = [curr_pos_labels[pos_ctr_iter] for curr_pos_labels in top_ce_ent_indices]
				
				curr_dataset = _get_dataset_from_tokenized_inputs(
					model_type=model_type,
					tokenized_inputs=tokenized_mentions,
					tokenized_labels=tokenized_entities,
					pos_label_idxs=curr_pos_label_idxs,
					neg_labels_idxs=neg_labels_idxs
				)
				all_datasets += [curr_dataset]
			
			# LOGGER.info("Intentional embed")
			# embed()
			dataset = ConcatDataset(all_datasets)
			return dataset
		
		elif neg_strategy in ["bienc_distill"]:
			
			assert model_type == "cross_enc", f"Model_type={model_type} not supported for neg_strategy={neg_strategy}"
			# Find top-num_labels labels for each mention using biencoder
			num_labels = distill_args["n_labels"]
			
			# Re-use get_hard_negs function to get top-scoring entities under biencoder model by passing list of empty list for pos_label_idxs
			labels_idxs, label_scores = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=tokenized_mentions,
				labels_tokens_list=tokenized_entities,
				pos_label_idxs=[[] for _ in range(len(tokenized_mentions))],
				num_negs=num_labels
			)
			
			# Pair each of top-num_labels labels with their mention
			_, paired_token_idxs = _get_paired_token_idxs(
				tokenized_inputs=tokenized_mentions,
				tokenized_labels=tokenized_entities,
				pos_label_idxs=[int(mention["label_id"]) for mention in mention_data],
				neg_labels_idxs=labels_idxs
			)
			paired_token_idxs = torch.cat(paired_token_idxs)
			
			label_scores = torch.Tensor(label_scores)
			dataset = TensorDataset(paired_token_idxs, label_scores)
			return dataset
	
		else:
			# Creating list of list type pos_label_idxs because get_hard_negs function expects this format
			pos_label_idxs = [[int(mention["label_id"])] for mention in mention_data]
			n_labels = len(tokenized_entities)
			if neg_strategy == "random":
				neg_labels_idxs = get_random_negs(data=mention_data, seed=0, num_negs=num_negs, n_labels=n_labels, label_key="label_id")
			elif neg_strategy == "bienc_hard_negs" and biencoder is None:
				warnings.warn("Mining negative randomly as biencoder model is not provided")
				neg_labels_idxs = get_random_negs(data=mention_data, seed=0, num_negs=num_negs, n_labels=n_labels, label_key="label_id")
			elif neg_strategy == "bienc_hard_negs" and biencoder is not None:
				neg_labels_idxs, _ = get_hard_negs_biencoder(
					biencoder=biencoder,
					input_tokens_list=tokenized_mentions,
					labels_tokens_list=tokenized_entities,
					pos_label_idxs=pos_label_idxs,
					num_negs=num_negs
				)
			elif neg_strategy == "tfidf_hard_negs":
				entity_file = nsw_args["entity_file"]
				neg_labels_idxs = get_hard_negs_tfidf(mentions_data=mention_data,
													  entity_file=entity_file,
													  pos_label_idxs=pos_label_idxs,
													  num_negs=num_negs)
			elif neg_strategy == "in_batch":
				neg_labels_idxs = []
			elif neg_strategy == "precomp":
				# Load num_negs+1 labels form this file so that if gt label is present in this list,
				# we can remove it and still have num_negs negatives
				ent_indices, _, _ = get_precomputed_ents_w_scores(
					ent_w_scores_file=ent_w_score_file,
					num_labels=num_negs+1,
					n_ments=len(tokenized_mentions),
					tokenized_entities=tokenized_entities
				)
				
				neg_labels_idxs = []
				# Remove pos-label from list of labels for each input/mention if it is present and finally keep num_negs labels
				for ment_idx, curr_pos_labels in enumerate(pos_label_idxs):
					temp_neg_labels_idxs = [curr_label_idx for curr_label_idx in ent_indices[ment_idx]
											if curr_label_idx not in curr_pos_labels][:num_negs]
					
					assert len(temp_neg_labels_idxs) > 0
					while len(temp_neg_labels_idxs) < num_negs:
						temp_neg_labels_idxs += temp_neg_labels_idxs
						# temp_neg_labels_idxs += [temp_neg_labels_idxs[-1]]*(num_negs - len(temp_neg_labels_idxs))
					
					neg_labels_idxs += [temp_neg_labels_idxs]
					
			else:
				raise NotImplementedError(f"Negative sampling strategy = {neg_strategy} not implemented")
			
			# Simplifying list of single-element-list to list format
			pos_label_idxs = [curr_pos_labels[0] for curr_pos_labels in pos_label_idxs]
			dataset = _get_dataset_from_tokenized_inputs(
				model_type=model_type,
				tokenized_inputs=tokenized_mentions,
				tokenized_labels=tokenized_entities,
				pos_label_idxs=pos_label_idxs,
				neg_labels_idxs=neg_labels_idxs
			)
			
			return dataset
	except Exception as e:
		LOGGER.info(f"Exception raised in data_process.get_ent_link_dataset() {str(e)}")
		embed()
		raise e


def get_ent_link_ce_dataset(
	model_type,
	tokenizer,
	raw_data,
	ent_tokens_file,
	biencoder,
	neg_strategy,
	max_input_len,
	max_label_len,
	ent_w_score_file,
	num_pos_labels_for_distill,
	num_negs,
	seed,
):
	"""
	Get dataset with tokenized data for training a model on precomputed mention-entity scores.
	:param raw_data:
	:param tokenizer
	:param ent_tokens_file:
	:param model_type:
	:param max_input_len
	:param max_label_len:
	:param neg_strategy:
	:param num_negs:
	:param biencoder:
	:param num_pos_labels_for_distill:
	:param seed:
	:param ent_w_score_file
	:return: Object of type TensorDataset
	"""
	try:
		assert model_type == "bi_enc", f"Model_type = {model_type} is not supported in get_ent_link_ce_dataset function"
		mention_data, (title2id, id2title, id2text, kb_id2local_id) = raw_data
		
		#################################### TOKENIZE MENTIONS AND ENTITIES ############################################
		LOGGER.info("Loading and tokenizing mentions")
		all_tokenized_mentions = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=max_input_len,)["ids"]
								for mention in tqdm(mention_data)]
		LOGGER.info("Finished tokenizing mentions")
		all_tokenized_mentions = torch.LongTensor(all_tokenized_mentions)
		
		
		if ent_tokens_file is not None and os.path.isfile(ent_tokens_file):
			LOGGER.info(f"Reading tokenized entities from file {ent_tokens_file}")
			tokenized_entities = torch.LongTensor(np.load(ent_tokens_file))
		else:
			LOGGER.info(f"Tokenizing {len(id2title)} entities")
			tokenized_entities = [ get_candidate_representation(candidate_title=id2title[ent_id],
																candidate_desc=id2text[ent_id],
																tokenizer=tokenizer,
																max_seq_length=max_label_len)["ids"]
									for ent_id in tqdm(sorted(id2title))]
			tokenized_entities = torch.LongTensor(tokenized_entities)
		
		
		################################## READ CROSSENCODER SCORES FROM FILE  #########################################
		LOGGER.info(f"Read scores from  file ={ent_w_score_file}")
		with open(ent_w_score_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			
			ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			ment_idxs = dump_dict["ment_idxs"] if "ment_idxs" in dump_dict else np.arange(ment_to_ent_scores.shape[0])
			
			# mention_data = dump_dict["test_data"]
			# entity_tokens_list = dump_dict["entity_tokens_list"]
			# arg_dict = dump_dict["arg_dict"]
		
		n_ments, n_ents = ment_to_ent_scores.shape
		tokenized_mentions = torch.LongTensor(mention_tokens_list)
		assert (len(entity_id_list) == 0) or (entity_id_list == np.arange(n_ents)).all(), f"len(entity_id_list) = {len(entity_id_list)} and it needs to be coupled with entity scores"
		assert (tokenized_mentions == all_tokenized_mentions[ment_idxs]).all(), f"Error in mention tokenization, tokenized_mentions.shape={tokenized_mentions.shape}, "
		
		LOGGER.info(f"Sorting scores for each mention")
		ent_indices = np.vstack([np.arange(n_ents) for _ in range(n_ments)]) # shape = n_ments, n_ents
		assert ent_indices.shape == ment_to_ent_scores.shape, f"ent_indices.shape = {ent_indices.shape} != ment_to_ent_scores.shape = {ment_to_ent_scores.shape}"

		sorted_ent_indices, sorted_ent_scores = _sort_by_score(indices=ent_indices, scores=ment_to_ent_scores)

		top_ent_indices = sorted_ent_indices[:, :num_pos_labels_for_distill]
		top_ent_scores = sorted_ent_scores[:, :num_pos_labels_for_distill]

		assert top_ent_indices.shape == (n_ments, num_pos_labels_for_distill), f"ent_indices shape = {top_ent_indices.shape} does not match n_ments, num_labels = {n_ments, num_pos_labels_for_distill}"
		assert top_ent_indices.shape == top_ent_scores.shape, f"Indices arrays shape = {top_ent_indices.shape} is different from score array shape = {top_ent_scores.shape}"


		################################################################################################################
		
		######################### GENERATE POSITIVE AND NEGATIVE LABELS FOR EACH DATAPOINT #############################
		
		"""
		Loss functions to implement
		
		1. ce w/ top-ce entities
		2. Triplet loss -
			for each mention,
				pair on top-k ce with num_negs negative entity - chosen randomly or from entities beyond top-k negatives
		
		"""
		LOGGER.info(f"Loading negs for strategy = {neg_strategy}")
		if neg_strategy in ["top_ce_match"]:
			tkn_labels_for_distill  = []
			for curr_top_ent_indices in top_ent_indices:
				# Accumulate tokenizations of top labels/entities for this mention
				curr_entities = [tokenized_entities[curr_ent_idx].unsqueeze(0) for curr_ent_idx in curr_top_ent_indices]
				tkn_labels_for_distill += [torch.cat(curr_entities).unsqueeze(0)]
		
			tkn_labels_for_distill = torch.cat(tkn_labels_for_distill) # Shape : num_mentions x num_top_ents_per_mention x entity_seq_len
			
			LOGGER.info(f"top_ent_scores.shape = {top_ent_scores.shape}")
			LOGGER.info(f"tokenized_mentions.shape = {tokenized_mentions.shape}")
			LOGGER.info(f"tkn_labels_for_distill.shape = {tkn_labels_for_distill.shape}")
			dataset = TensorDataset(tokenized_mentions, tkn_labels_for_distill, top_ent_scores)
			return dataset
		
		elif neg_strategy in ["top_ce_w_bienc_hard_negs_trp", "top_ce_w_rand_negs_trp"]:
			
			# Triplet style - Find num_pos_labels_for_distill negatives and then pair each with one positive label (i.e. label with topk-ce scores)
			
			if biencoder is None or neg_strategy == "top_ce_w_rand_negs_trp":
				warnings.warn(f"Mining negative randomly as biencoder model is not provided or neg_strategy = {neg_strategy}")
				neg_labels_idxs = get_random_negs_w_blacklist(
					n_data=len(ment_idxs),
					seed=seed,
					num_negs=num_pos_labels_for_distill,
					n_labels=len(tokenized_entities),
					label_blacklist=top_ent_indices
				)
			else:
				# Get hard negatives for biencoder while treating top-cross-encoder labels as positive labels
				neg_labels_idxs, _ = get_hard_negs_biencoder(
					biencoder=biencoder,
					input_tokens_list=tokenized_mentions,
					labels_tokens_list=tokenized_entities,
					pos_label_idxs=top_ent_indices,
					num_negs=num_pos_labels_for_distill
				)
			
			trp_ment_tokens = []
			trp_pos_tokens = []
			trp_neg_tokens = []
			for ment_iter in range(n_ments):
				for label_iter in range(num_pos_labels_for_distill):
					curr_pos_ent_idx  = top_ent_indices[ment_iter][label_iter]
					curr_neg_ent_idx  = neg_labels_idxs[ment_iter][label_iter]
					
		
					trp_ment_tokens += [tokenized_mentions[ment_iter]]
					trp_pos_tokens += [tokenized_entities[curr_pos_ent_idx]]
					trp_neg_tokens += [tokenized_entities[curr_neg_ent_idx]]
				
			
			trp_ment_tokens = torch.stack(trp_ment_tokens) # shape: num_mentions*num_pos_labels_for_distill, seq_len
			trp_pos_tokens = torch.stack(trp_pos_tokens)   # shape: num_mentions*num_pos_labels_for_distill, seq_len
			trp_neg_tokens = torch.stack(trp_neg_tokens)   # shape: num_mentions*num_pos_labels_for_distill, seq_len
			
			# shape: num_mentions*num_pos_labels_for_distill, 1,  seq_len -
			# This allows us to use same biencoder forward function as used when using larger number of negatives per mention and training biencoder with ground-truth entity data
			trp_neg_tokens = trp_neg_tokens.unsqueeze(1)
			
			LOGGER.info(f"trp_ment_tokens.shape = {trp_ment_tokens.shape}")
			LOGGER.info(f"trp_pos_tokens.shape = {trp_pos_tokens.shape}")
			LOGGER.info(f"trp_neg_tokens.shape = {trp_neg_tokens.shape}")
		
			dataset = TensorDataset(trp_ment_tokens, trp_pos_tokens, trp_neg_tokens)
			
			return dataset
			
		elif neg_strategy in ["top_ce_w_bienc_hard_negs_ml", "top_ce_w_rand_negs_ml"]:
			LOGGER.info(f"Loading data for knowledge distillation from cross-encoder model w/ neg_strategy = {neg_strategy} and num_negs_per_pos = {num_negs}")
			
			# Multi-label style - Find num_negs negatives and pair with corresponding pos labels in multi-label fashion
			if biencoder is None or neg_strategy == "top_ce_w_rand_negs_ml":
				warnings.warn(f"Mining negative randomly as biencoder model is not provided or neg_strategy = {neg_strategy}")
				neg_labels_idxs = get_random_negs_w_blacklist(
					n_data=len(ment_idxs),
					seed=seed,
					num_negs=num_negs,
					n_labels=len(tokenized_entities),
					label_blacklist=top_ent_indices.cpu().numpy()
				)
			else:
				# Get hard negatives for biencoder while treating top-cross-encoder labels as positive labels
				neg_labels_idxs, _ = get_hard_negs_biencoder(
					biencoder=biencoder,
					input_tokens_list=tokenized_mentions,
					labels_tokens_list=tokenized_entities,
					pos_label_idxs=top_ent_indices,
					num_negs=num_negs
				)
			
			# top_ce_ent_indices.shape == n_data, num_labels_for_distill
			# neg_labels_idxs.shape == n_data, num_labels_for_distill
		
			dataset = _get_dataset_from_tokenized_inputs_multi_label(
				model_type=model_type,
				tokenized_inputs=tokenized_mentions,
				tokenized_labels=tokenized_entities,
				pos_labels_idxs=top_ent_indices,
				neg_labels_idxs=neg_labels_idxs
			)
			
			# LOGGER.info("Intentional embed")
			# embed()
			return dataset
		

		else:
			raise NotImplementedError(f"neg_strategy = {neg_strategy} not supported in get_ent_link_ce_dataset()")
			
	except Exception as e:
		LOGGER.info(f"Exception raised in data_process.get_ent_link_dataset() {str(e)}")
		embed()
		raise e


def get_xmc_dataset(model_type, tokenizer, input_data, labels, max_input_len, max_label_len, biencoder, neg_strategy, num_negs,
					pos_strategy, max_pos_labels, total_labels_per_input, precomp_fname):
	"""
	Get dataset with tokenized data for XMC from input_data
	It first tokenizes the dataset and then creates a dataset with positive/negative training datapoints
	:param model_type:
	:param input_data:
	:param labels:
	:param tokenizer:
	:param max_input_len:
	:param max_label_len:
	:param neg_labels:
	:return:
	"""
	try:
		############################################# Tokenize all the labels ##########################################
		LOGGER.info("Tokenizing labels")
		processed_labels = [tokenize_input(input=label,
									  tokenizer=tokenizer,
									  max_seq_length=max_label_len
									  )["token_idxs"]
							for label in tqdm(labels)]
		processed_labels = torch.LongTensor(processed_labels)
		################################################################################################################
		
		########################################## Tokenize the input data #############################################
		LOGGER.info("Tokenizing inputs")
		processed_inputs = []
		for datapoint in tqdm(input_data):
			input_tokens = tokenize_input(input=datapoint["input"],
										  tokenizer=tokenizer,
										  max_seq_length=max_input_len
										  )["token_idxs"]
			
			processed_inputs += [input_tokens]
		processed_inputs = torch.LongTensor(processed_inputs)
		################################################################################################################
		
		pos_label_idxs = [datapoint["label_idxs"] for datapoint in input_data]
		
		###################################### Find hard negatives for each datapoint ##################################
		n_labels = len(labels)
		if neg_strategy == "random":
			neg_labels_idxs = get_random_negs(data=input_data, seed=0, num_negs=num_negs, n_labels=n_labels, label_key="label_idxs")
		elif neg_strategy == "bienc_hard_negs" and biencoder is None:
			warnings.warn("Mining negative randomly as biencoder model is not provided")
			neg_labels_idxs = get_random_negs(data=input_data, seed=0, num_negs=num_negs, n_labels=n_labels, label_key="label_idxs")
		elif neg_strategy == "bienc_hard_negs" and biencoder is not None:
			neg_labels_idxs, _ = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=processed_inputs,
				labels_tokens_list=processed_labels,
				pos_label_idxs=pos_label_idxs,
  				num_negs=num_negs
			)
		elif neg_strategy == "in_batch":
			neg_labels_idxs = []
		elif neg_strategy == "precomp":
			# fname = "../../results/SiameseXML++/Astec/1_LF-AmazonTitles-131K/bow/v_default_params_0/trn_predictions_clf.npz"
			pred_label_mat = load_npz(precomp_fname)
			neg_labels_idxs = []
			n_max_pos_labels = max([len(x) for x in pos_label_idxs])
			
			# Retrieve num_negs + n_max_pos_labels top labels for each datapoint so that we have at least
			# num_negs labels for each datapoint after removing positive labels from the retrieved set
			topk_pred_label_idxs = _get_topk_from_sparse(X=pred_label_mat, k=num_negs+n_max_pos_labels)
			
			for data_ctr in range(len(processed_inputs)):
				
				curr_pred_label_idxs = topk_pred_label_idxs[data_ctr]
				
				# Filter out an positive label from current top labels and we will use them as negatives
				curr_neg_label_idxs = [idx for idx in curr_pred_label_idxs if idx not in pos_label_idxs[data_ctr]]
				

				assert len(curr_neg_label_idxs) >= num_negs, f"len(curr_neg_label_idxs) = {len(curr_neg_label_idxs)} should be larger than or equal to num_negs = {num_negs}"
				
				neg_labels_idxs += [curr_neg_label_idxs[:num_negs]]
				
			neg_labels_idxs = np.array(neg_labels_idxs)
		else:
			raise NotImplementedError(f"Negative sampling strategy = {neg_strategy} not implemented")
		################################################################################################################
		
		if pos_strategy == "flatten_pos":
			
			# Flatten positive labels for each datapoint and also create as many copies of each datapoint
			pos_label_idxs = [pos_label for curr_pos_labels in pos_label_idxs for pos_label in curr_pos_labels]
			
			# Create num_pos_label copies of neg_labels and input_tokens
			neg_labels_idxs = [curr_neg_labels for (curr_neg_labels, datapoint) in zip(neg_labels_idxs, input_data)
								for _ in range(len(datapoint["label_idxs"])) ]
			neg_labels_idxs = np.array(neg_labels_idxs)
			processed_inputs = [input_tokens for (input_tokens, datapoint) in zip(processed_inputs, input_data) for _ in range(len(datapoint["label_idxs"]))]
			processed_inputs = torch.stack(processed_inputs)
			
			LOGGER.info(f"Creating dataset for model = {model_type}")
			dataset = _get_dataset_from_tokenized_inputs(
				model_type=model_type,
				tokenized_inputs=processed_inputs,
				tokenized_labels=processed_labels,
				pos_label_idxs=pos_label_idxs,
				neg_labels_idxs=neg_labels_idxs
			)
			return dataset
		elif pos_strategy == "keep_pos_together":
			assert num_negs >= total_labels_per_input - 1, f"Num negs mined initially = {num_negs} but should be at least total_labels_per_input - 1 = {total_labels_per_input - 1}"
			n_inputs = len(input_data)
			
			
			all_inputs = []
			all_labels_idxs = []
			all_label_vals = [] # stores 0/1 value for labels for each input indicated whether the label is a pos or neg label
			for input_ctr in range(n_inputs):
				
				# If datapoint has more than max_pos_labels, then split it across multiple inputs,
				# each containing up to max_pos_labels positive labels and followed by negative labels.
				n_splits = int(np.ceil(len(pos_label_idxs[input_ctr])/max_pos_labels))
				
				for pos_ctr in range(n_splits):
					curr_split_pos_labels = np.array(pos_label_idxs[input_ctr][pos_ctr*n_splits: (pos_ctr+1)*n_splits])
					labels_for_curr_input = np.concatenate((curr_split_pos_labels, neg_labels_idxs[input_ctr]))
				
					gt_label_vals = [1]*len(curr_split_pos_labels) + [0]*len(neg_labels_idxs[input_ctr])
					
					labels_for_curr_input = labels_for_curr_input[:total_labels_per_input]
					gt_label_vals_for_curr_input = gt_label_vals[:total_labels_per_input]
					
					assert len(labels_for_curr_input) == total_labels_per_input, f"len(labels_for_curr_input) = {len(labels_for_curr_input)} != total_labels_per_input = {total_labels_per_input}"
					assert len(gt_label_vals_for_curr_input) == total_labels_per_input, f"len(gt_label_vals_for_curr_input) = {len(gt_label_vals_for_curr_input)} != total_labels_per_input = {total_labels_per_input}"
					
					all_inputs += [processed_inputs[input_ctr]]
					all_labels_idxs += [labels_for_curr_input]
					all_label_vals += [gt_label_vals_for_curr_input]
				
			
			LOGGER.info(f"Original number of datapoints = {len(processed_inputs)}")
			LOGGER.info(f"Number of datapoints after creating batches = {len(all_inputs)}\n (this is larger than value above as some inputs are duplicated due to large number of positive associated with it) ")
			all_inputs = torch.stack(all_inputs)
			all_labels_idxs = torch.tensor(np.stack(all_labels_idxs))
			all_label_vals = torch.tensor(all_label_vals)
			
			
			assert model_type == "cross_enc", f"model_type other than cross-encoder not supported"
			

			all_tknzd_input_label_pairs = []
			for idx, input_tkns in enumerate(all_inputs):
				# Create paired rep for current mentions with all negative entities/labels
				curr_pair_reps = []
				for curr_label_idx in all_labels_idxs[idx]:
					curr_pair = create_input_label_pair(input_token_idxs=input_tkns, label_token_idxs=processed_labels[curr_label_idx])
					curr_pair_reps += [curr_pair.unsqueeze(0)]
				
				curr_pair_reps = torch.cat(curr_pair_reps)
				all_tknzd_input_label_pairs += [curr_pair_reps.unsqueeze(0)]
			
			all_tknzd_input_label_pairs = torch.cat(all_tknzd_input_label_pairs)
		

			return TensorDataset(all_tknzd_input_label_pairs, all_label_vals)
			
		else:
			raise NotImplementedError(f"Pos_strategy = {pos_strategy} not implemented")
		
	except Exception as e:
		embed()
		raise e

def _get_dataset_from_tokenized_inputs(model_type, tokenized_inputs, tokenized_labels, pos_label_idxs, neg_labels_idxs):
	"""
	Helper function that creates dataset containing positive/negative examples
	using already tokenized labels and inputs.
	
	:param model_type:
	:param tokenized_inputs:
	:param tokenized_labels:
	:param pos_label_idxs: List of positive labels for corresponding input
	:param neg_labels_idxs: List of which contains list of negative labels for each input
	:return: Object of type TensorDataset
	"""

	try:
		LOGGER.info(f"Shape of tokenized_labels = {tokenized_labels.shape}")
		
		
		
		if model_type == "bi_enc":
			tokenized_pos_label = tokenized_labels[pos_label_idxs]
			if len(neg_labels_idxs) == 0:
				tokenized_tensor_data = TensorDataset(tokenized_inputs, tokenized_pos_label)
			else:
				tokenized_neg_labels  = []
				for idx in range(len(tokenized_inputs)):
					# Accumulate tokenizations of neg labels/entities for this mention
					curr_neg_labels = [tokenized_labels[neg_idx].unsqueeze(0) for neg_idx in neg_labels_idxs[idx]]
					tokenized_neg_labels += [torch.cat(curr_neg_labels).unsqueeze(0)]
	
				tokenized_neg_labels = torch.cat(tokenized_neg_labels) # Shape : num_mentions x num_neg_per_mention x entity_len
				tokenized_tensor_data = TensorDataset(tokenized_inputs, tokenized_pos_label, tokenized_neg_labels)
			
		elif model_type == "cross_enc":
			
			pos_paired_token_idxs, neg_paired_token_idxs = _get_paired_token_idxs(tokenized_inputs, tokenized_labels,
																				  pos_label_idxs, neg_labels_idxs)
			pos_paired_token_idxs = torch.cat(pos_paired_token_idxs)
			neg_paired_token_idxs = torch.cat(neg_paired_token_idxs)
			tokenized_tensor_data = TensorDataset(pos_paired_token_idxs, neg_paired_token_idxs)
		else:
			raise NotImplementedError(f"Data loading for model_type = {model_type} not supported.")
		
	
		return tokenized_tensor_data
	except Exception as e:
		embed()
		raise e



def _get_tokenized_labels(list_of_label_idxs, all_tokenized_labels):
	
	tokenized_labels  = []
	for curr_label_idxs in list_of_label_idxs:
		# Accumulate tokenizations of labels/entities for this mention/input
		curr_tknzd_labels = [all_tokenized_labels[idx].unsqueeze(0) for idx in curr_label_idxs]
		tokenized_labels += [torch.cat(curr_tknzd_labels).unsqueeze(0)]

	tokenized_labels = torch.cat(tokenized_labels) # Shape : num_mentions x num_pos_per_mention x entity_seq_len
	
	return tokenized_labels




def _get_dataset_from_tokenized_inputs_multi_label(model_type, tokenized_inputs, tokenized_labels, pos_labels_idxs, neg_labels_idxs):
	"""
	Helper function that creates dataset containing positive/negative examples
	using already tokenized labels and inputs.
	
	:param model_type:
	:param tokenized_inputs:
	:param tokenized_labels:
	:param pos_labels_idxs: List which contains list of positive labels for each input
	:param neg_labels_idxs: List which contains list of negative labels for each input
	:return: Object of type TensorDataset
	"""

	try:
		LOGGER.info(f"Shape of tokenized_labels = {tokenized_labels.shape}")
		if model_type == "bi_enc":
			
			assert len(neg_labels_idxs) != 0, f"in-batch negs for multi-label not supported"
			
			tokenized_pos_labels  = _get_tokenized_labels(list_of_label_idxs=pos_labels_idxs, all_tokenized_labels=tokenized_labels)
			tokenized_neg_labels  = _get_tokenized_labels(list_of_label_idxs=neg_labels_idxs, all_tokenized_labels=tokenized_labels)
			tokenized_tensor_data = TensorDataset(tokenized_inputs, tokenized_pos_labels, tokenized_neg_labels)
		else:
			raise NotImplementedError(f"Data loading for model_type = {model_type} not supported.")
		
	
		return tokenized_tensor_data
	except Exception as e:
		embed()
		raise e


def _get_paired_token_idxs(tokenized_inputs, tokenized_labels, pos_label_idxs, neg_labels_idxs):
	"""
	Concatenates input and label tokens using pos_label_idxs to create a tensor of pos_paired_token_idxs, and
	Concatenates input and label tokens using neg_label_idxs to create a tensor of neg_paired_token_idxs
	:param tokenized_inputs:
	:param tokenized_labels:
	:param pos_label_idxs: List of positive label per input
	:param neg_labels_idxs: List of list of negative labels per input
	:return: List of input-paired-with-pos-labels, and list of input-paired-with-neg-labels
	"""
	
	tokenized_pos_label = tokenized_labels[pos_label_idxs]
	pos_paired_token_idxs = []
	neg_paired_token_idxs = []
	for idx, input_tkns in enumerate(tokenized_inputs):
		pos_label_tkns = tokenized_pos_label[idx]
		# Create paired rep for current mentions with positive/ground-truth entities/labels
		pos_pair_rep = create_input_label_pair(input_token_idxs=input_tkns, label_token_idxs=pos_label_tkns)
		pos_paired_token_idxs += [pos_pair_rep.unsqueeze(0)]
		
		# Create paired rep for current mentions with all negative entities/labels
		curr_neg_pair_reps = []
		for neg_idx in neg_labels_idxs[idx]:
			curr_neg_pair = create_input_label_pair(input_token_idxs=input_tkns, label_token_idxs=tokenized_labels[neg_idx])
			curr_neg_pair_reps += [curr_neg_pair.unsqueeze(0)]
		
		curr_neg_pair_reps = torch.cat(curr_neg_pair_reps)
		neg_paired_token_idxs += [curr_neg_pair_reps.unsqueeze(0)]
	
	return pos_paired_token_idxs, neg_paired_token_idxs
	

def create_input_label_pair(input_token_idxs, label_token_idxs):
	"""
	Remove cls token from label (this is the first token) and concatenate with input
	:param input_token_idxs:
	:param label_token_idxs:
	:return:
	"""
	if isinstance(input_token_idxs, torch.Tensor):
		return torch.cat((input_token_idxs, label_token_idxs[1:]))
	else: # numpy arrays support concat with + operator
		return input_token_idxs + label_token_idxs[1:]


class XMCTensorDataset(TensorDataset):
	r"""Dataset wrapping input tensors, label tensors along with positive labels for each input.

	Each sample will be retrieved by indexing tensors along the first dimension.
	pos_labels is a sparse matrix with # columns = # labels with 1 at index (i,j) indicating
	that label j is present for ith datapoint.
	
	Args:
		pos_labels : Scipy sparse matrix of shape N x L. N: Number of data-points, L: Number of labels
		*tensors (Tensor): tensors that have the same size of the first dimension.
	"""
	
	def __init__(self, pos_labels: object, *tensors: object) -> None:
		super(XMCTensorDataset, self).__init__(*tensors)
		
		assert isinstance(pos_labels, csr_matrix)
		assert pos_labels.shape[0] == len(self), f"Number of rows in pos_labels matrix = {pos_labels.shape} " \
												 f"does not match dataset size = {len(self)}"
		
		self.pos_labels = pos_labels
	
	def __getitem__(self, index):
		return tuple(tensor[index] for tensor in self.tensors) + (self.pos_labels[index].indices,)
	
	def __len__(self):
		return self.tensors[0].size(0)


class XMCDataset(object):
	
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels


class NSWDataset(object):
	
	def __init__(self, all_pos_neg_pairs, all_pos_neg_pair_token_idxs, total_paths, n_ments, n_ents):
		self.all_pos_neg_pairs = all_pos_neg_pairs
		self.all_pos_neg_pair_token_idxs = all_pos_neg_pair_token_idxs
		self.total_paths = total_paths
		self.n_ments = n_ments
		self.n_ents = n_ents
	
	def __len__(self):
		return len(self.all_pos_neg_pair_token_idxs)
	
	@staticmethod
	def concat(all_datasets):
		total_paths = 0
		total_n_ents = 0
		total_n_ments = 0
		all_pos_neg_pairs = []
		all_pos_neg_pair_token_idxs = []
		for dataset in all_datasets:
			assert isinstance(dataset, NSWDataset)
			all_pos_neg_pairs += dataset.all_pos_neg_pairs
			all_pos_neg_pair_token_idxs += [dataset.all_pos_neg_pair_token_idxs]
			total_paths += dataset.total_paths
			total_n_ments += dataset.n_ments
			total_n_ents += dataset.n_ents
		
		all_pos_neg_pair_token_idxs = ConcatDataset(all_pos_neg_pair_token_idxs)
		return NSWDataset(all_pos_neg_pairs=all_pos_neg_pairs,
						  all_pos_neg_pair_token_idxs=all_pos_neg_pair_token_idxs,
						  n_ments=total_n_ments,
						  n_ents=total_n_ents,
						  total_paths=total_paths)
	
	@staticmethod
	def concat_nested_data_format(all_datasets):
		all_path_pos_neg_pairs, all_path_pos_neg_pair_token_idxs = [], []
		total_n_ments = 0
		total_n_ents = 0
		total_paths = 0
		for dataset in all_datasets:
			assert isinstance(dataset, NSWDataset)
			all_path_pos_neg_pairs += dataset.all_pos_neg_pairs
			all_path_pos_neg_pair_token_idxs += dataset.all_pos_neg_pair_token_idxs
			total_paths += dataset.total_paths
			total_n_ments += dataset.n_ments
			total_n_ents += dataset.n_ents
			
		return NSWDataset(all_pos_neg_pairs=all_path_pos_neg_pairs,
						  all_pos_neg_pair_token_idxs=all_path_pos_neg_pair_token_idxs,
						  n_ments=total_n_ments,
						  n_ents=total_n_ents,
						  total_paths=total_paths)

######## Function from blink/biencoder.data_process.py ######
def get_context_representation(
	sample,
	tokenizer,
	max_seq_length,
	mention_key="mention",
	context_key="context",
	ent_start_token=ENT_START_TAG,
	ent_end_token=ENT_END_TAG,
):
	
	mention_tokens = []
	if sample[mention_key] and len(sample[mention_key]) > 0:
		mention_tokens = tokenizer.tokenize(sample[mention_key])
		mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

	context_left = sample[context_key + "_left"]
	context_right = sample[context_key + "_right"]
	context_left = tokenizer.tokenize(context_left)
	context_right = tokenizer.tokenize(context_right)

	left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
	right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
	left_add = len(context_left)
	right_add = len(context_right)
	if left_add <= left_quota:
		if right_add > right_quota:
			right_quota += left_quota - left_add
	else:
		if right_add <= right_quota:
			left_quota += right_quota - right_add

	context_tokens = (
		context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
	)

	context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
	input_ids = tokenizer.convert_tokens_to_ids(context_tokens)[:max_seq_length]
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	assert len(input_ids) == max_seq_length, f"Input_ids len = {len(input_ids)} != max_seq_len ({max_seq_length})"

	return {
		"tokens": context_tokens,
		"ids": input_ids,
	}


def get_candidate_representation(
	candidate_desc,
	tokenizer,
	max_seq_length,
	candidate_title=None,
	title_tag=ENT_TITLE_TAG,
):
	try:
		cls_token = tokenizer.cls_token
		sep_token = tokenizer.sep_token
		cand_tokens = tokenizer.tokenize(candidate_desc)
		if candidate_title is not None:
			title_tokens = tokenizer.tokenize(candidate_title)
			cand_tokens = title_tokens + [title_tag] + cand_tokens
	
		cand_tokens = cand_tokens[: max_seq_length - 2]
		cand_tokens = [cls_token] + cand_tokens + [sep_token]
	
		input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		assert len(input_ids) == max_seq_length
	
		return {
			"tokens": cand_tokens,
			"ids": input_ids,
		}
	except Exception as e:
		embed()
		raise e


### Extra Code


def __xmc_collate_fn(batch_data):
	"""
	Covert list of tuples to tuple of list. Each final list has batch_size number of elements.
	:param batch_data: List of tuples of size batch_size.
	:return: Collated batch data.
	"""
	
	batch_input, batch_labels, batch_label_idx, batch_pos_labels = zip(*batch_data)
	
	batch_input = torch.stack(batch_input, dim=0)
	batch_labels = torch.stack(batch_labels, dim=0)
	batch_label_idx = torch.stack(batch_label_idx, dim=0)
	
	return batch_input, batch_labels, batch_label_idx, batch_pos_labels


def __process_xmc_data_for_biencoder(
		input_data,
		labels,
		tokenizer,
		max_input_length,
		max_label_length,
):
	"""
	
	:param input_data: List of raw input datapoints. Each datapoint is dict with keys `input` and `label_idxs`
	:param labels: List of raw labels.
	:param tokenizer:
	:param max_input_length:
	:param max_label_length:
	
	# TODO: Fix docstring
	:return: Dict mapping to list of input tokens, corresponding label tokens etc
	"""
	try:
		############################################# Tokenize all the labels ##########################################
		processed_labels = []
		for idx, label in enumerate(tqdm(labels)):
			label_tokens = tokenize_input(input=label,
										  tokenizer=tokenizer,
										  max_seq_length=max_label_length
										  )
			processed_labels.append(label_tokens["token_idxs"])
		
		################################################################################################################
		
		########################################## Tokenize the input data #############################################
		processed_data = []
		for idx, datapoint in enumerate(tqdm(input_data)):
			input_tokens = tokenize_input(input=datapoint["input"],
										  tokenizer=tokenizer,
										  max_seq_length=max_input_length
										  )
			label_idxs = datapoint["label_idxs"]
			
			# Create a separate (input, label) record for each label corresponding to given input
			for curr_label_idx in label_idxs:
				record = {"input": input_tokens["token_idxs"],
						  "label": processed_labels[curr_label_idx],
						  "label_idx": [curr_label_idx],
						  "all_label_idxs": label_idxs}
				
				processed_data.append(record)
		
		input_token_idxs = torch.tensor([x["input"] for x in processed_data], dtype=torch.long)
		label_token_idxs = torch.tensor([x["label"] for x in processed_data], dtype=torch.long)
		label_idxs = torch.tensor([x["label_idx"] for x in processed_data], dtype=torch.long)
		all_label_idxs = [x["all_label_idxs"] for x in processed_data]
		
		rows_and_cols = [(i, label_i) for i, curr_pos_labels in enumerate(all_label_idxs)
						 for label_i in curr_pos_labels]
		rows, cols = zip(*rows_and_cols)
		N = len(processed_data)
		L = len(processed_labels)
		_data = np.ones(len(rows))
		pos_labels = csr_matrix((_data, (rows, cols)), shape=(N, L))
		
		data = {
			"input_token_idxs": input_token_idxs,
			"label_token_idxs": label_token_idxs,
			"label_idxs": label_idxs,
			"pos_labels": pos_labels
		}
		
		tensor_data = XMCTensorDataset(input_token_idxs, label_token_idxs, label_idxs, pos_labels)
		return data, tensor_data
	except Exception as e:
		embed()
		raise e


def __process_xmc_data_for_cross_enc(
		input_data,
		labels,
		tokenizer,
		max_input_length,
		max_label_length,
		neg_labels
):
	"""
	
	:param input_data: List of raw input datapoints. Each datapoint is dict with keys `input` and `label_idxs`
	:param labels: List of raw labels.
	:param tokenizer:
	:param max_input_length:
	:param max_label_length:
	:param neg_labels: List of negative labels for each datapoint
	
	# TODO: Fix docstring
	:return: Dict mapping to list of input tokens, corresponding label tokens etc
	"""
	try:
		# Tokenize all labels
		processed_labels = []
		for idx, label in enumerate(tqdm(labels)):
			label_tokens = tokenize_input(input=label,
										  tokenizer=tokenizer,
										  max_seq_length=max_label_length
										  )
			processed_labels.append(label_tokens["token_idxs"])
		
		# Tokenize all inputs
		processed_data = []
		for idx, datapoint in enumerate(tqdm(input_data)):
			input_tokens = tokenize_input(input=datapoint["input"],
										  tokenizer=tokenizer,
										  max_seq_length=max_input_length
										  )
			label_idxs = datapoint["label_idxs"]
			
			# Create a separate (input, label) record for each positive label corresponding to given input
			for curr_label_idx in label_idxs:
				record = {"input_label_pair": create_input_label_pair(input_tokens["token_idxs"],
																	  processed_labels[curr_label_idx]),
						  "label_idx": [curr_label_idx],
						  "score": 1}  # 1 as this is a positive input,label pair
				
				processed_data.append(record)
				
				for neg_idx in neg_labels[idx]:
					record = {"input_label_pair": create_input_label_pair(input_tokens["token_idxs"],
																		  processed_labels[neg_idx]),
							  "label_idx": [neg_idx],
							  "score": 0}  # 0 as this is a negative input,label pair
					
					processed_data.append(record)
		
		
		paired_token_idxs = torch.tensor([x["input_label_pair"] for x in processed_data], dtype=torch.long)
		label_idxs = torch.tensor([x["label_idx"] for x in processed_data], dtype=torch.long)
		scores = torch.tensor([x["score"] for x in processed_data], dtype=torch.float)
		
		data = {
			"paired_input_label_token_idxs": paired_token_idxs,
			"label_idxs": label_idxs,
			"scores": scores
		}
		
		tensor_data = TensorDataset(paired_token_idxs, label_idxs, scores)
		return data, tensor_data
	except Exception as e:
		embed()
		raise e
