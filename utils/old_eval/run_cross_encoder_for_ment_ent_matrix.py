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

import wandb
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from blink.biencoder.biencoder import load_biencoder
# from blink.crossencoder.crossencoder import load_crossencoder

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



def create_paired_dataset(encoded_mentions, encoded_entities, max_pair_length, batch_size, num_pairs_per_input):
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
		if len(curr_batch) == num_pairs_per_input:
			paired_dataset += [curr_batch]
			curr_batch = []

	if len(curr_batch) > 0:
		# padded_curr_batch = np.zeros( np.array(paired_dataset[-1]).shape, dtype=int)
		padded_curr_batch = np.zeros( (num_pairs_per_input, len(curr_batch[0])), dtype=int)
		padded_curr_batch[:len(curr_batch), :] = curr_batch
		paired_dataset += [padded_curr_batch]

	paired_dataset = torch.LongTensor(paired_dataset)

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
		
		wandb.log({"batch_idx": step,
				   "frac_done": float(step)/len(dataloader)})
		all_scores += [batch_score]

	return all_scores



def main(data_id, n_ment, batch_size):
	try:
		pretrained_dir = "../../BLINK_models"
		data_dir = "../../data/BLINK_data"
		res_dir = "../../results/2_CURApproxEval"
		
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
		
		wandb.init(project=f"EntityLinking~CE_Ment2EntMatrix", dir=res_dir, config={"n_ment":n_ment,
																					"batch_size":batch_size,
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
		
		LOGGER.info("Loading test samples")
		test_data = _get_test_samples(test_filename=dataset["filename"],
										  wikipedia_id2local_id=wikipedia_id2local_id,
										  logger=LOGGER,
										  )
		test_data = test_data[:n_ment]
		
		LOGGER.info("Tokenize test samples")
		# First extract all mentions and tokenize them
		mention_tokens_list = prepare_crossencoder_mentions(tokenizer=tokenizer,
																samples=test_data,
																max_context_length=max_ment_length)
		
		complete_entity_tokens_list = np.load(f"{pretrained_dir}/entity_tokens_{len(id2title)}_{max_ent_length}.npy")
		
		try:
			entity_id_list = json.load(open(f"{res_dir}/bienc_nns_{dataset['name']}_{len(mention_tokens_list)}.json", "r"))
			LOGGER.info(f"Number of entities = {len(entity_id_list)}!!!")
		except FileNotFoundError:
			LOGGER.info("File with entity list not found. Runnig with first 101 entities!!!")
			entity_id_list = np.arange(101)
			
		entity_tokens_list = complete_entity_tokens_list[entity_id_list]
		n_ent = len(entity_tokens_list)
	
		with torch.no_grad():
			
			LOGGER.info(f"Computing score for each test entity with each mention. batch_size={batch_size}, n_ment={n_ment}, n_ent={n_ent}")
			
			dataloader = create_paired_dataset(encoded_mentions=mention_tokens_list,
											   encoded_entities=entity_tokens_list,
											   max_pair_length=max_pair_length,
											   batch_size=1,
											   num_pairs_per_input=batch_size)
			
			LOGGER.info("Running cross encoder model now")
			crossencoder.eval()
			_ment_to_ent_scores = _run_cross_encoder(cross_encoder=crossencoder,
													 dataloader=dataloader,
													 max_ment_length=max_ment_length)
			
			
			_ment_to_ent_scores = torch.stack([x.view(-1) for x in _ment_to_ent_scores]).view(-1)
			_ment_to_ent_scores = _ment_to_ent_scores[:n_ment*n_ent] # Remove scores for padded data
			ment_to_ent_scores = _ment_to_ent_scores.view(n_ment, n_ent).cpu()
			
			pickle.dump((ment_to_ent_scores, test_data, mention_tokens_list, entity_id_list, entity_tokens_list),
						open(f"{res_dir}/ment_to_ent_scores_{dataset['name']}_n_m_{n_ment}_n_e_{n_ent}.pkl", "wb"))
		
		LOGGER.info("Done")
	except KeyboardInterrupt:
		LOGGER.info("Interrupted by keyboard")
		embed()
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()



if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Compute mention x entity matrix using cross-encoder model')
	
	parser.add_argument("--data_id", type=int, default=4, help="Dataset id")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions")
	parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
	
	args = parser.parse_args()
	
	_n_ment = args.n_ment
	_data_id = args.data_id
	_batch_size = args.batch_size

	main(data_id=_data_id, n_ment=_n_ment, batch_size=_batch_size)
