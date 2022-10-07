import os
import sys
import json
import time
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
import pickle

import wandb
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from blink.biencoder.biencoder import load_biencoder
# from blink.crossencoder.crossencoder import load_crossencoder
#
# from blink.crossencoder.data_process import prepare_crossencoder_mentions
# from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer



logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

def _load_entities(
	entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
	# only load candidate encoding if not using faiss index
	if faiss_index is None:
		candidate_encoding = torch.load(entity_encoding)
		indexer = None
	else:
		if logger:
			logger.info("Using faiss index to retrieve entities.")
		candidate_encoding = None
		assert index_path is not None, "Error! Empty indexer path."
		if faiss_index == "flat":
			indexer = DenseFlatIndexer(1)
		elif faiss_index == "hnsw":
			indexer = DenseHNSWFlatIndexer(1)
		else:
			raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
		indexer.deserialize_from(index_path)

	# load all the 5903527 entities
	title2id = {}
	id2title = {}
	id2text = {}
	wikipedia_id2local_id = {}
	local_idx = 0
	with open(entity_catalogue, "r") as fin:
		lines = fin.readlines()
		for line in lines:
			entity = json.loads(line)

			if "idx" in entity:
				split = entity["idx"].split("curid=")
				if len(split) > 1:
					wikipedia_id = int(split[-1].strip())
				else:
					wikipedia_id = entity["idx"].strip()

				assert wikipedia_id not in wikipedia_id2local_id
				wikipedia_id2local_id[wikipedia_id] = local_idx

			title2id[entity["title"]] = local_idx
			id2title[local_idx] = entity["title"]
			id2text[local_idx] = entity["text"]
			local_idx += 1
	return (
		candidate_encoding,
		title2id,
		id2title,
		id2text,
		wikipedia_id2local_id,
		indexer,
	)


def load_models(args, logger=None):

	# load biencoder model
	if logger:
		logger.info("loading biencoder model")
	with open(args.biencoder_config) as json_file:
		biencoder_params = json.load(json_file)
		biencoder_params["path_to_model"] = args.biencoder_model
	biencoder = load_biencoder(biencoder_params)

	crossencoder = None
	crossencoder_params = None
	if not args.fast:
		# load crossencoder model
		if logger:
			logger.info("loading crossencoder model")
		with open(args.crossencoder_config) as json_file:
			crossencoder_params = json.load(json_file)
			crossencoder_params["path_to_model"] = args.crossencoder_model
		crossencoder = load_crossencoder(crossencoder_params)

	# load candidate entities
	if logger:
		logger.info("loading candidate entities")
	(
		candidate_encoding,
		title2id,
		id2title,
		id2text,
		wikipedia_id2local_id,
		faiss_indexer,
	) = _load_entities(
		args.entity_catalogue,
		args.entity_encoding,
		faiss_index=getattr(args, 'faiss_index', None),
		index_path=getattr(args, 'index_path' , None),
		logger=logger,
	)

	return (
		biencoder,
		biencoder_params,
		crossencoder,
		crossencoder_params,
		candidate_encoding,
		title2id,
		id2title,
		id2text,
		wikipedia_id2local_id,
		faiss_indexer,
	)


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
	test_samples = []
	with open(test_filename, "r") as fin:
		lines = fin.readlines()
		for line in lines:
			record = json.loads(line)
			record["label"] = str(record["label_id"])

			# for tac kbp we should use a separate knowledge source to get the entity id (label_id)
			if kb2id and len(kb2id) > 0:
				if record["label"] in kb2id:
					record["label_id"] = kb2id[record["label"]]
				else:
					continue

			# check that each entity id (label_id) is in the entity collection
			elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
				try:
					key = int(record["label"].strip())
					if key in wikipedia_id2local_id:
						record["label_id"] = wikipedia_id2local_id[key]
					else:
						continue
				except:
					continue

			# LOWERCASE EVERYTHING !
			record["context_left"] = record["context_left"].lower()
			record["context_right"] = record["context_right"].lower()
			record["mention"] = record["mention"].lower()
			test_samples.append(record)

	if logger:
		logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
	return test_samples


def _get_test_samples(
	test_filename, wikipedia_id2local_id, logger
):
	kb2id = None
	test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
	return test_samples



def create_paired_dataset(encoded_mentions, encoded_entities, max_pair_length, batch_size, pair_batch_size):
	"""
	Create a list of representations by pairing each mention with each entity
	:param encoded_mentions:
	:param encoded_entities:
	:param max_pair_length:
	:return:
	"""
	paired_dataset = []

	def get_pairs():
		for mention in encoded_mentions:
			for ent in encoded_entities:
				pair = np.concatenate((mention, ent[1:]))
				yield pair[:max_pair_length]

	curr_batch = []
	for pair in get_pairs():
		curr_batch += [pair]
		if len(curr_batch) == pair_batch_size:
			paired_dataset += [curr_batch]
			curr_batch = []

	if len(curr_batch) > 0:
		paired_dataset += [curr_batch]

	paired_dataset = torch.tensor(paired_dataset)

	tensor_data = TensorDataset(paired_dataset)
	sampler = SequentialSampler(tensor_data)
	dataloader = DataLoader(
		tensor_data, sampler=sampler, batch_size=batch_size
	)
	return dataloader

def _run_cross_encoder(cross_encoder, dataloader, max_ment_length):
	"""
	Run cross-encoder model on given data and return scores
	:param cross_encoder:
	:param dataloader:
	:param max_ment_length:
	:return:
	"""
	all_scores = []
	for step, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
		batch_input, = batch
		batch_input = batch_input.to(cross_encoder.device)
		batch_score = cross_encoder.score_candidate(batch_input, first_segment_end=max_ment_length)

		all_scores += [batch_score]

	#TODO: Make sure format of all_scores is correct
	return all_scores



def _get_cross_enc_pred(crossencoder, max_ment_length, max_pair_length, batch_ment_tokens, complete_entity_tokens_list, batch_retrieved_indices):
		
	
		# Create pair of input,nnbr entity tokens. Strip off first token from nnbr entity as that is CLS token and also limit to max_pair_length
		batch_nnbr_ent_tokens = [complete_entity_tokens_list[nnbr_indices] for nnbr_indices in batch_retrieved_indices.cpu().data.numpy()]
		
		batch_paired_inputs = []
		for i, ment_tokens in enumerate(batch_ment_tokens):
			paired_inputs = torch.stack([torch.cat((ment_tokens.view(-1), nnbr[1:]))[:max_pair_length] for nnbr in batch_nnbr_ent_tokens[i]])
			batch_paired_inputs += [paired_inputs]

		batch_paired_inputs = torch.stack(batch_paired_inputs).to(crossencoder.device)

		batch_crossenc_scores = crossencoder.score_candidate(batch_paired_inputs, first_segment_end=max_ment_length)
		
		batch_crossenc_scores = batch_crossenc_scores.to(batch_retrieved_indices.device)
		
		# Since we just have nearest nbr entities, we need to map argmax in crossenc_scores to original ent id
		batch_crossenc_pred = batch_retrieved_indices.gather(1, torch.argmax(batch_crossenc_scores, dim=1, keepdim=True))
		batch_crossenc_pred = batch_crossenc_pred.view(-1)
		
		return batch_crossenc_pred

# def compute_anchor_portions():
#
# 	rng = np.random.default_rng(seed=0)
# 	k_ment = 100
# 	k_ent = 50
# 	n_ment = len(mention_tokens_list)
# 	n_ent = len(entity_tokens_list)
# 	# Choose a subset of mentions as anchors
# 	anchor_mention_idxs = sorted(rng.choice(np.arange(n_ment), size=k_ment, replace=False))
# 	anchor_mentions = mention_tokens_list[anchor_mention_idxs]
#
# 	# Choose a subset of entities as anchors
# 	anchor_entity_idxs = sorted(rng.choice(np.arange(n_ent), size=k_ent, replace=False))
# 	anchor_entities = entity_tokens_list[anchor_entity_idxs]
#
# 	print(f"Number of anchor mentions {k_ment}/{n_ment}")
# 	print(f"Number of anchor entities {k_ent}/{len(entity_id_list)}")
#
# 	print("Computing score for each test entity with each anchor mention")
# 	batch_size=crossencoder_params["eval_batch_size"]
# 	pair_batch_size = 50
# 	dataloader = create_paired_dataset(encoded_mentions=anchor_mentions,
# 										   encoded_entities=entity_tokens_list,
# 										   max_pair_length=max_pair_length,
# 										   batch_size=batch_size,
# 										   pair_batch_size=pair_batch_size)
#
# 	with torch.no_grad():
# 		crossencoder.eval()
# 		_anc_ment_to_ent_scores = _run_cross_encoder(cross_encoder=crossencoder,
# 														dataloader=dataloader,
# 														max_ment_length=max_ment_length)
#
#
# 	print("Computing score for each test mention with each anchor entity")
# 	batch_size=crossencoder_params["eval_batch_size"]
# 	pair_batch_size = 50
# 	# Compute score for each test mention with each anchor entity
# 	dataloader = create_paired_dataset(encoded_mentions=mention_tokens_list,
# 										   encoded_entities=anchor_entities,
# 										   max_pair_length=max_pair_length,
# 										   batch_size=batch_size,
# 										   pair_batch_size=pair_batch_size)
#
#
# 	with torch.no_grad():
# 		crossencoder.eval()
# 		_ment_to_anc_ent_scores = _run_cross_encoder(cross_encoder=crossencoder,
# 														dataloader=dataloader,
# 														max_ment_length=max_ment_length)
#
# 	anc_ment_to_ent_scores = torch.stack([x.view(-1) for x in _anc_ment_to_ent_scores]).view(-1).view(len(anchor_mention_idxs), len(entity_tokens_list))
# 	ment_to_anc_ent_scores = torch.stack([x.view(-1) for x in _ment_to_anc_ent_scores]).view(-1).view(len(mention_tokens_list), len(anchor_entity_idxs))
#
# 	print("Anchor ment to Ent Scores", anc_ment_to_ent_scores.shape)
# 	print("Mention to anchor ent scores", ment_to_anc_ent_scores.shape)
#
# 	pickle.dump((anc_ment_to_ent_scores, anchor_mention_idxs, mention_tokens_list), open(f"{res_dir}/anc_ment_to_ent_scores_{dataset['name']}_{len(mention_tokens_list)}_{len(entity_tokens_list)}_{k_ment}_{k_ent}.pkl", "wb"))
# 	pickle.dump((ment_to_anc_ent_scores, anchor_entity_idxs, entity_id_list, entity_tokens_list), open(f"{res_dir}/ment_to_anc_ent_scores_{dataset['name']}_{len(mention_tokens_list)}_{len(entity_tokens_list)}_{k_ment}_{k_ent}.pkl", "wb"))
#
def main(data_id, n_ment, batch_size, top_k):
	try:
		assert top_k > 1
		pretrained_dir = "../../BLINK_models"
		data_dir = "../../data/BLINK_data"
		res_dir = "../../results/3_CrossEnc"
		rng = np.random.default_rng(seed=0)
		
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		
		DATASETS = [
			{
				"name": "AIDA-YAGO2 testa",
				"filename": f"{data_dir}/BLINK_benchmark/AIDA-YAGO2_testa.jsonl",
			},
			{
				"name": "AIDA-YAGO2 testb",
				"filename": f"{data_dir}/BLINK_benchmark/AIDA-YAGO2_testb.jsonl",
			},
			{"name": "ACE 2004", "filename": f"{data_dir}/BLINK_benchmark/ace2004_questions.jsonl"},
			{"name": "aquaint", "filename": f"{data_dir}/BLINK_benchmark/aquaint_questions.jsonl"},
			{
				"name": "clueweb - WNED-CWEB (CWEB)",
				"filename": f"{data_dir}/BLINK_benchmark/clueweb_questions.jsonl",
			},
			{"name": "msnbc", "filename": f"{data_dir}/BLINK_benchmark/msnbc_questions.jsonl"},
			{
				"name": "wikipedia - WNED-WIKI (WIKI)",
				"filename": f"{data_dir}/BLINK_benchmark/wnedwiki_questions.jsonl",
			},
		]
		dataset = DATASETS[data_id]
		
		wandb.init(project=f"EntityLinking~CE_w_BiEnc_Retriever", dir=res_dir, config={"n_ment":n_ment,
																					   "batch_size":batch_size,
																					   "top_k":top_k,
																					   "data_id":data_id,
																					   "data_name":dataset['name'],
																					   "CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]})
		PARAMETERS = {
			"faiss_index": None,
			"index_path": None,
			"test_entities": None,
			"test_mentions": None,
			"interactive": False,
			"biencoder_model": f"{pretrained_dir}/biencoder_wiki_large.bin",
			"biencoder_config": f"{pretrained_dir}/biencoder_wiki_large.json",
			"entity_catalogue": f"{pretrained_dir}/entity.jsonl",
			"entity_encoding": f"{pretrained_dir}/all_entities_large.t7",
			"crossencoder_model": f"{pretrained_dir}/crossencoder_wiki_large.bin",
			"crossencoder_config": f"{pretrained_dir}/crossencoder_wiki_large.json",
			"output_path": "output",
			"fast": False,
			"top_k": 100,
		}
		args = argparse.Namespace(**PARAMETERS)
		
		(
			biencoder,
			biencoder_params,
			crossencoder,
			crossencoder_params,
			candidate_encoding,
			title2id,
			id2title,
			id2text,
			wikipedia_id2local_id,
			faiss_indexer,
		) = load_models(args, LOGGER)
		
		tokenizer = crossencoder.tokenizer
		max_ent_length = 128
		max_ment_length = 32
		max_pair_length = 160
		
		
		test_data = _get_test_samples(test_filename=dataset["filename"],
										  wikipedia_id2local_id=wikipedia_id2local_id,
										  logger=LOGGER,
										  )
		test_data = test_data[:n_ment]
		# First extract all mentions and tokenize them
		mention_tokens_list = prepare_crossencoder_mentions(tokenizer=tokenizer,
																samples=test_data,
																max_context_length=max_ment_length)
		
		complete_entity_tokens_list = np.load(f"{pretrained_dir}/entity_tokens_{len(id2title)}_{max_ent_length}.npy")
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list).to(biencoder.device)
		
		candidate_encoding = candidate_encoding.t() # Take transpose for easier matrix multiplication ops later
		
		curr_mentions_tensor = torch.tensor(mention_tokens_list)
		curr_gt_labels = np.array([x["label_id"] for x in test_data])
		
		batched_data = TensorDataset(curr_mentions_tensor, torch.LongTensor(curr_gt_labels))
		sampler = SequentialSampler(batched_data)
		bienc_dataloader = DataLoader(batched_data, sampler=sampler, batch_size=batch_size)
		
		bienc_pred_labels  = []
		crossenc_pred_w_rand_retrvr_labels = []
		crossenc_pred_w_bienc_retrvr_labels = []
		with torch.no_grad():
			biencoder.eval()
			crossencoder.eval()
			
			LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment}, top_k={top_k}")
			LOGGER.info(f"Bi encoder model device {biencoder.device}")
			LOGGER.info(f"Cross encoder model device {crossencoder.device}")
			for batch_idx, (batch_ment_tokens, batch_ment_gt_labels) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
				batch_ment_tokens =  batch_ment_tokens.to(biencoder.device)
				
				ment_encodings = biencoder.encode_context(batch_ment_tokens)
				batch_bienc_scores = ment_encodings.mm(candidate_encoding)
				
				# batch_bienc_scores = biencoder.score_candidate(batch_ment_tokens, None, cand_encs=candidate_encoding)
				batch_bienc_pred = torch.argmax(batch_bienc_scores, dim=1)
				
				wandb.log({"bienc_batch_idx": batch_idx,
						   "bienc_frac_done": float(batch_idx)/len(bienc_dataloader)})
				
				# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
				# If it were complete mention-entity matrix then we would have to use ment_idx
				batch_bienc_top_k_scores, batch_bienc_top_k_indices = batch_bienc_scores.topk(top_k)
				
			
				batch_rand_indices = torch.tensor(np.array([rng.choice(np.arange(len(complete_entity_tokens_list)), size=top_k-1, replace=False)
															for _ in range(batch_size)]))
			
				# Add ground-truth entity to randomly chosen indices
				batch_rand_indices = torch.cat((batch_rand_indices, batch_ment_gt_labels.view(-1, 1)), dim=1)
				
				_get_cross_enc_pred_w_retrvr = lambda retrv_indices : _get_cross_enc_pred(crossencoder=crossencoder,
																						  max_pair_length=max_pair_length,
																						  max_ment_length=max_ment_length,
																						  batch_ment_tokens=batch_ment_tokens,
																						  complete_entity_tokens_list=complete_entity_tokens_list,
																						  batch_retrieved_indices=retrv_indices)
				
				# Use cross-encoder for re-ranking bi-encoder and randomly sampled entities
				batch_crossenc_pred_w_bienc_retrvr = _get_cross_enc_pred_w_retrvr(batch_bienc_top_k_indices)
				batch_crossenc_pred_w_rand_retrvr = _get_cross_enc_pred_w_retrvr(batch_rand_indices)
				
				
				bienc_pred_labels.extend(batch_bienc_pred.cpu().numpy().tolist())
				crossenc_pred_w_rand_retrvr_labels.extend(batch_crossenc_pred_w_rand_retrvr.cpu().numpy().tolist())
				crossenc_pred_w_bienc_retrvr_labels.extend(batch_crossenc_pred_w_bienc_retrvr.cpu().numpy().tolist())
				
				wandb.log({"crossenc_batch_idx": batch_idx,
						   "crossenc_frac_done": float(batch_idx)/len(bienc_dataloader)})

				
			# for ment_idx, ment_tokens in tqdm(enumerate(curr_mentions_tensor), position=0, leave=True):
			#
			# 	bienc_scores = biencoder.score_candidate(ment_tokens, None, cand_encs=candidate_encoding)
			# 	bienc_scores = bienc_scores.view(-1)
			# 	bienc_pred = int(torch.argmax(bienc_scores))
			#
			# 	# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
			# 	# If it were complete mention-entity matrix then we would have to use ment_idx
			#
			# 	bienc_top_k_scores, bienc_top_k_indices = bienc_scores.topk(top_k)
			# 	bienc_top_k_indices = bienc_top_k_indices.data.numpy()
			#
			# 	# Create pair of input,nnbr entity tokens. Strip off first token from nnbr entity as that is CLS token and also limit to max_pair_length
			# 	nnbr_ent_tokens = complete_entity_tokens_list[bienc_top_k_indices]
			# 	paired_inputs = torch.stack([torch.cat((ment_tokens.view(-1), nnbr[1:]))[:max_pair_length] for nnbr in nnbr_ent_tokens])
			#
			# 	paired_inputs = paired_inputs.unsqueeze(0)
			# 	crossenc_scores = crossencoder.score_candidate(paired_inputs, first_segment_end=max_ment_length)
			#
			# 	# Since we just have nearest nbr entities, we need to map argmax in crossenc_scores to original ent id
			# 	crossenc_pred = bienc_top_k_indices[int(torch.argmax(crossenc_scores))]
			#
			# 	bienc_pred_labels += [bienc_pred]
			# 	crossenc_pred_labels += [crossenc_pred]
			#
		
		bienc_acc = np.mean(bienc_pred_labels == curr_gt_labels)
		crossenc_acc_w_bienc_acc = np.mean(crossenc_pred_w_bienc_retrvr_labels == curr_gt_labels)
		crossenc_acc_w_rand_acc = np.mean(crossenc_pred_w_rand_retrvr_labels == curr_gt_labels)
		LOGGER.info(f"Bi-Encoder Accuracy = {bienc_acc}")
		LOGGER.info(f"Cross-Encoder Accuracy (w/ bienc_retriever) = {crossenc_acc_w_bienc_acc}")
		LOGGER.info(f"Cross-Encoder Accuracy (w/ rand retriever)  = {crossenc_acc_w_rand_acc}")
		json.dump({"bienc_acc":bienc_acc,
					"crossenc_acc_w_bienc": crossenc_acc_w_bienc_acc,
					"crossenc_acc_w_rand": crossenc_acc_w_rand_acc},
				  
				  open(f"{res_dir}/res_{dataset['name']}_m={n_ment}_k={top_k}_{batch_size}.json", "w"),
				  indent=4)
		
		
		LOGGER.info("Done")
	except KeyboardInterrupt:
		LOGGER.info("Interrupted by keyboard")
		embed()
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		time.sleep(2)



if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	
	parser.add_argument("--data_id", type=int, default=4, help="Dataset id")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
	
	args = parser.parse_args()
	
	_n_ment = args.n_ment
	_top_k = args.top_k
	_batch_size = args.batch_size
	_data_id = args.data_id
	
	
	main(data_id=_data_id, n_ment=_n_ment, top_k=_top_k, batch_size=_batch_size)
