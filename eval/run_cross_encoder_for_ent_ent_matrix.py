import os
import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
import pickle

import faiss
import wandb
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers.tokenization_bert import BertTokenizer
from eval.nsw_eval_zeshel import compute_ent_embeds
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from utils.data_process import load_entities, get_candidate_representation
from models.params import ENT_START_TAG, ENT_END_TAG

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_ent_tokens_as_ments(ent_file, max_seq_len, bert_model_type="bert-base-uncased", use_lowercase=True):
	LOGGER.info(f"Tokenizing entities from file {ent_file}")
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=ent_file)
	
	tokenizer = BertTokenizer.from_pretrained(bert_model_type, do_lower_case=use_lowercase)
	
	tokenized_entities = []
	for ent_id in tqdm(sorted(id2title)):
		candidate_title = id2title[ent_id]
		candidate_desc = id2title[ent_id]
		
		cls_token = tokenizer.cls_token
		sep_token = tokenizer.sep_token
		cand_tokens = tokenizer.tokenize(candidate_desc)
		
		title_tokens = tokenizer.tokenize(candidate_title)
		cand_tokens = [ENT_START_TAG] + title_tokens + [ENT_END_TAG] + cand_tokens
	
		cand_tokens = cand_tokens[: max_seq_len - 2]
		cand_tokens = [cls_token] + cand_tokens + [sep_token]
	
		input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
		padding = [0] * (max_seq_len - len(input_ids))
		input_ids += padding
		assert len(input_ids) == max_seq_len
	
		tokenized_entities += [input_ids]
		
	return tokenized_entities


def get_ent_pair_dataset(topk_ents, token_opt, entity_id_list_x, tokenized_entities, tokenized_entities_as_ments):
	all_ent_pairs = []
	for curr_ent_idx, curr_topk_ents in zip(entity_id_list_x, topk_ents):
		if token_opt == "e2e":
			pairs = [np.concatenate((tokenized_entities[curr_ent_idx], tokenized_entities[nbr_ent][1:])) for nbr_ent in curr_topk_ents]
		elif token_opt == "m2e":
			pairs = [np.concatenate((tokenized_entities_as_ments[curr_ent_idx], tokenized_entities[nbr_ent][1:])) for nbr_ent in curr_topk_ents]
		else:
			raise NotImplementedError(f"Token_opt = {token_opt} not supported")
		
		all_ent_pairs += pairs
		
	all_ent_pairs_dataset = TensorDataset(torch.LongTensor(all_ent_pairs))
	
	return all_ent_pairs_dataset


def get_ents_for_scoring(ent_embeds, query_embeds, topk):
	d = ent_embeds.shape[-1]
	LOGGER.info(f"Finding nearest entities for all mentions using embed of dim = {d}")
	index = faiss.IndexFlatIP(d)
	index.add(ent_embeds)
	topk_ents = index.search(query_embeds, k=topk)[1]
	
	return topk_ents

def run(biencoder, crossencoder, data_fname, n_ent_x_start, n_ent_x_arg, n_ent_y_arg, topk, batch_size, dataset_name, res_dir, embed_type, token_opt, misc, arg_dict, topk_ents_file):
	"""
	
	:param biencoder:
	:param crossencoder:
	:param data_fname:
	:param n_ent_x_arg:
	:param n_ent_y_arg:
	:param topk:
	:param batch_size:
	:param dataset_name:
	:param res_dir:
	:param embed_type:
	:param token_opt:
	:param misc:
	:param arg_dict:
	:return:
	"""
	if biencoder: biencoder.eval()
	crossencoder.eval()
	if crossencoder.device == torch.device("cpu"):
		wandb.alert(title="No GPUs found", text=f"{crossencoder.device}")
		raise Exception("No GPUs found!!!")
	try:
		
		LOGGER.info(f"Computing score for each test entity with each mention. batch_size={batch_size}, n_ent_x={n_ent_x_arg}, n_ent_x={n_ent_y_arg}, topk = {topk}")
		# Load entity tokens
		tokenized_entities = np.load(data_fname["ent_tokens_file"])
		n_ent_x = len(tokenized_entities) if n_ent_x_arg < 0 else n_ent_x_arg
		n_ent_y = len(tokenized_entities) if n_ent_y_arg < 0 else n_ent_y_arg
		
		entity_id_list_x = n_ent_x_start + np.arange(n_ent_x)
		entity_id_list_y = np.arange(n_ent_y)

		# Compute/load entity embeddings
		if embed_type == "anchor":
			with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
				dump_dict = pickle.load(fin)
				crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				loaded_entity_id_list = dump_dict["entity_id_list"]
			if torch.is_tensor(crossenc_ment_to_ent_scores):
				crossenc_ment_to_ent_scores = crossenc_ment_to_ent_scores.cpu().detach().numpy()
			
			dim = crossenc_ment_to_ent_scores.shape[0]
			max_ent_x_y = max(n_ent_x, n_ent_y)
			ent_embeds = np.zeros((max_ent_x_y, dim))
			
			if len(loaded_entity_id_list) != max_ent_x_y:
				LOGGER.info("\n\n Mismatch b/w ")
				LOGGER.info(f"len(loaded_entity_id_list) ={len(loaded_entity_id_list)}, max_n_ent_x_y= {max_ent_x_y}\n\n")
				
			# Fill in entity embeddings as per entity ids in loaded_entity_id_list
			loaded_ent_embeds = np.ascontiguousarray(np.transpose(crossenc_ment_to_ent_scores))
			relevant_indices = loaded_entity_id_list[np.where(loaded_entity_id_list < max_ent_x_y)]
			relevant_ent_embeds = loaded_ent_embeds[np.where(loaded_entity_id_list < max_ent_x_y)]
			
			ent_embeds[relevant_indices] = relevant_ent_embeds
			ent_embeds = ent_embeds.astype(dtype=np.float32)
		elif embed_type in ["bienc", "tfidf"]:
			ent_embeds = compute_ent_embeds(
				embed_type=embed_type,
				biencoder=biencoder,
				entity_tokens_file=data_fname["ent_tokens_file"],
				entity_file=data_fname["ent_file"],
			)
		elif embed_type == "none":
			ent_embeds = []
		else:
			raise NotImplementedError(f"embed_type={embed_type} not supported")
	
		with torch.no_grad():
			LOGGER.info(f"Running score computation with {n_ent_x} x {n_ent_y} entities!!!")
			
			
			# Find top-k entities to score for each entity using cross-encoder
			# rng = np.random.default_rng(seed=0)
			# topk_ents = np.array([rng.integers(0, n_ent_y, size=topk) for _ in range(n_ent_x)])
			
			if os.path.isfile(topk_ents_file):
				LOGGER.info(f"Loading topk ents from file - {topk_ents_file}")
				with open(topk_ents_file, "r") as fin:
					topk_ents_dict = json.load(fin)
				topk_ents = topk_ents_dict[str(topk)]
				# Repeat top-k ents as many times as the number of entity rows i.e. n_ent_x
				topk_ents = np.array([topk_ents for _ in range(n_ent_x)])
			else:
				LOGGER.info("Finding topk ents for each entity")
				topk_ents = get_ents_for_scoring(ent_embeds=ent_embeds[entity_id_list_y], query_embeds=ent_embeds[entity_id_list_x], topk=topk)
				assert n_ent_y_arg < 0, "topk_ents indices returned should be mapped to proper entity indices " \
										"if n_ent_y_arg >= 0 because it means that we are not searching over " \
										"all entities so topk_ents returned from get_ents_for_scoring are relative to " \
										"ent_embeds given to the function and not relative to ALL entities "
			
			if token_opt == "e2e":
				all_ent_pairs_dataset = get_ent_pair_dataset(
					topk_ents=topk_ents,
					token_opt=token_opt,
					entity_id_list_x=entity_id_list_x,
					tokenized_entities=tokenized_entities,
					tokenized_entities_as_ments=None
				)
			elif token_opt == "m2e":
				tokenized_entities_as_ments = get_ent_tokens_as_ments(
					ent_file=data_fname["ent_file"],
					max_seq_len=MAX_PAIR_LENGTH,
				)
				all_ent_pairs_dataset = get_ent_pair_dataset(
					topk_ents=topk_ents,
					token_opt=token_opt,
					entity_id_list_x=entity_id_list_x,
					tokenized_entities=tokenized_entities,
					tokenized_entities_as_ments=tokenized_entities_as_ments
				)
			else:
				raise NotImplementedError(f"Token_opt = {token_opt} not supported")
			
			
			
			dataloader = DataLoader(all_ent_pairs_dataset, batch_size=batch_size, shuffle=False)
			
			LOGGER.info("Running cross encoder model now")
			all_scores_list = []
			for step, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
				batch_input, = batch
				batch_input = batch_input.to(crossencoder.device)
				
				batch_score = crossencoder.score_candidate(batch_input, first_segment_end=MAX_ENT_LENGTH) # Context len = MAX_ENT_LENGTH as we are concatenating two entities
				all_scores_list += [batch_score]
				
				wandb.log({"batch_idx": step,
						   "frac_done": float(step)/len(dataloader)})
			
			
			all_scores = torch.cat(all_scores_list)
			all_scores = all_scores[:n_ent_x*topk] # Remove scores for padded data to get shape = (n_ent*topk,)
			ent_to_ent_scores = all_scores.view(n_ent_x, topk).cpu() # shape: n_ent x topk
			
			LOGGER.info(f"Computed score matrix of shape = {ent_to_ent_scores.shape}")
			
			# embed()
			curr_res_dir = f"{res_dir}/{dataset_name}"
			Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
			with open(f"{curr_res_dir}/ent_to_ent_scores_n_e_{n_ent_x}x{n_ent_y}_topk_{topk}_embed_{embed_type}_{token_opt}_{misc}.pkl", "wb") as fout:
				res = {
					"ent_to_ent_scores":ent_to_ent_scores,
					"ent_to_ent_scores.shape":ent_to_ent_scores.shape,
					"topk_ents":topk_ents,
					"n_ent_x": n_ent_x,
					"n_ent_y": n_ent_y,
					"token_opt": token_opt,
					"entity_id_list_x":entity_id_list_x,
					"entity_id_list_y":entity_id_list_y,
					"entity_tokens_list":tokenized_entities,
					"arg_dict":arg_dict
				}
				pickle.dump(res, fout)
		
		LOGGER.info("Done")
	
	except KeyboardInterrupt as e:
		embed()
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def main():
	exp_id = "Zeshel_Ent2Ent"
	data_dir = "../../data/zeshel"
	
	
	worlds = get_zeshel_world_info()
	
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model for computing mention-entity scoring matrix')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ent_x_start", type=int, default=0, help="Start offset for n_ent_x")
	parser.add_argument("--n_ent_x", type=int, default=-1, help="Number of entities in x dim of entity-entity matrix.  -1 for all entities")
	parser.add_argument("--n_ent_y", type=int, default=-1, help="Number of entities in y dim of entity-entity matrix.  -1 for all entities")
	parser.add_argument("--topk", type=int, default=10, help="Number of entities to score for each entity")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	parser.add_argument("--embed_type", type=str, choices=["tfidf", "bienc", "anchor", "none"], required=True, help="Type of embeddings to use for retrieving top-k entities for scoring w/ cross-enc. If None then no either prefixed entities are chosen using topk_file or entities are chosen at random")
	parser.add_argument("--token_opt", type=str, choices=["e2e", "m2e"], required=True, help="How to tokenize each entity in entity pair. "
																							 "m2e tokenizes first entity like a mention w/ ent start and end tokens")
	
	parser.add_argument("--topk_ents_file", type=str, default="", help="File containing entities to compute each entity against")
	parser.add_argument("--res_dir", type=str, required=True, help="Directory to save results")
	parser.add_argument("--bi_model_file", type=str, default="", help="Biencoder Model config or ckpt file")
	parser.add_argument("--cross_model_file", type=str, required=True, help="Crossencoder Model config or ckpt file")
	
	parser.add_argument("--misc", type=str, default="", help="misc suffix for output file")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	args = parser.parse_args()

	data_name = args.data_name
	topk = args.topk
	n_ent_x_start = args.n_ent_x_start
	n_ent_x = args.n_ent_x
	n_ent_y = args.n_ent_y
	batch_size = args.batch_size
	embed_type = args.embed_type
	token_opt = args.token_opt
	res_dir = args.res_dir
	bi_model_file = args.bi_model_file
	cross_model_file = args.cross_model_file
	topk_ents_file = args.topk_ents_file
	disable_wandb = args.disable_wandb
	
	
	misc = args.misc
	
	arg_dict = args.__dict__
	
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=res_dir)
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	# Load models
	if cross_model_file.endswith(".json"):
		with open(cross_model_file, "r") as fin:
			config = json.load(fin)
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
		
	
	if os.path.isfile(bi_model_file):
		LOGGER.info(f"Loading biencoder from bienc file = {bi_model_file}")
		if bi_model_file.endswith(".json"):
			with open(bi_model_file, "r") as fin:
				config = json.load(fin)
				biencoder = BiEncoderWrapper.load_model(config=config)
		else:
			biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	else:
		biencoder = None
		LOGGER.info(f"Not loading biencoder as bienc file = {bi_model_file} does not exist")
	
	config = {
			"goal": "Compute entity-entity pairwise similarity matrix",
			"batch_size":batch_size,
			"n_ent_x":n_ent_x,
			"n_ent_y":n_ent_y,
			"embed_type":embed_type,
			"topk":topk,
			"data_name":data_name,
			"misc":misc,
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(arg_dict)
	
	try:
		wandb.init(
			project=exp_id,
			dir=res_dir,
			config=config,
			mode="disabled" if disable_wandb else "online"
		)
	except:
		try:
			wandb.init(
				project=exp_id,
				dir=res_dir,
				config=config,
				settings=wandb.Settings(start_method="fork"),
				mode="disabled" if disable_wandb else "online"
			)
		
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(
				project=exp_id,
				dir=res_dir,
				config=config,
				mode="offline",
			)
	
	# assert n_ent_x_start == 0 or (embed_type != "anchor" and (not os.path.isfile(topk_ents_file)) )
	assert n_ent_x_start == 0 or (embed_type != "anchor")
	
	iter_worlds = worlds if data_name == "all" else [("", data_name)]
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		run(
			biencoder=biencoder,
			crossencoder=crossencoder,
			data_fname=DATASETS[world_name],
			n_ent_x_start=n_ent_x_start,
			n_ent_x_arg=n_ent_x,
			n_ent_y_arg=n_ent_y,
			topk=topk,
			token_opt=token_opt,
			embed_type=embed_type,
			batch_size=batch_size,
			dataset_name=data_name,
			res_dir=res_dir,
			misc=misc,
			arg_dict=arg_dict,
			topk_ents_file=topk_ents_file
		)
	


if __name__ == "__main__":
	main()
