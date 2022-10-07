import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path

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


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
	biencoder.eval()
	nns = []
	candidate_encoding = candidate_encoding.t()
	for batch in tqdm(dataloader, position=0, leave=True):
		batch_input, = batch
		batch_input = batch_input.to(biencoder.device)
		
		with torch.no_grad():
			if indexer is not None:
				batch_encoding = biencoder.encode_context(batch_input).numpy()
				batch_encoding = np.ascontiguousarray(batch_encoding)
				scores, indices = indexer.search_knn(batch_encoding, top_k)
			else:
				ment_encodings = biencoder.encode_context(batch_input)
				batch_bienc_scores = ment_encodings.mm(candidate_encoding)
				# scores = biencoder.score_candidate(
				# 	batch_input, None, cand_encs=candidate_encoding  # .to(device)
				# )
				scores, indices = batch_bienc_scores.topk(top_k)
				indices = indices.data.numpy()

		nns.extend(indices)
	return nns



def main(data_id, n_ment):
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
		
		tokenizer = biencoder.tokenizer
		max_ent_length = 128
		max_ment_length = 32
		max_pair_length = 160
		
		
		test_data = _get_test_samples(test_filename=dataset["filename"],
										  wikipedia_id2local_id=wikipedia_id2local_id,
										  logger=LOGGER,
										  )
		# First extract all mentions and tokenize them
		mention_tokens_list = prepare_crossencoder_mentions(tokenizer=tokenizer,
																samples=test_data,
																max_context_length=max_ment_length)
		
		assert len(sys.argv) >= 2, "num_ments argument not passed"
		
		
		mention_tokens_list = mention_tokens_list[:n_ment]
		
		with torch.no_grad():
			biencoder.eval()
			curr_mentions_tensor = torch.tensor(mention_tokens_list).unsqueeze(1)
			sampler = SequentialSampler(curr_mentions_tensor)
			bienc_dataloader = DataLoader(curr_mentions_tensor, sampler=sampler, batch_size=1)
			binenc_nns = _run_biencoder(biencoder=biencoder,
										 dataloader=bienc_dataloader,
										 candidate_encoding=candidate_encoding)
		
		
			all_nns = list(set(np.array(binenc_nns).reshape(-1).tolist()))
			
			print(f"Number of entities in neighborhood {len(all_nns)}")
			
			all_gt_labels = [x["label_id"] for x in test_data[:n_ment]]
			all_labels = sorted(list(set(all_nns + all_gt_labels)))
			
			print(f"Number of entities in neighborhood + gt entities {len(all_labels)}")
			json.dump(all_labels, open(f"{res_dir}/bienc_nns_{dataset['name']}_{len(mention_tokens_list)}.json", "w"))
	except Exception as e:
		embed()


if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions")
	parser.add_argument("--data_id", type=int, default=4, help="Dataset id")
	
	args = parser.parse_args()
	
	_n_ment = args.n_ment
	_data_id = args.data_id
	
	main(n_ment=_n_ment, data_id=_data_id)
