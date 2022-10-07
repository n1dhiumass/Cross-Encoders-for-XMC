import os
import sys
import json
import torch
import pickle
import logging
import argparse



from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np



from eval.eval_utils import compute_label_embeddings, compute_input_embeddings

from models.biencoder import BiEncoderWrapper


from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from utils.data_process import load_entities, load_mentions, get_context_representation


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def run(biencoder, res_dir, data_info, n_ment, batch_size, misc, arg_dict, ):
	
	try:
		biencoder.eval()
		data_name, data_fname = data_info
	
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = biencoder.tokenizer
		
		LOGGER.info("Loading test samples")
		test_data = load_mentions(mention_file=data_fname["ment_file"],
								  kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[:n_ment] if n_ment > 0 else test_data
		
		LOGGER.info(f"Tokenize {n_ment} test samples")
		# First extract all mentions and tokenize them
		mention_tokens_list = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=MAX_MENT_LENGTH)["ids"]
								for mention in tqdm(test_data)]
		
		mention_tokens_list = torch.LongTensor(mention_tokens_list)
		complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		
		n_ent = len(complete_entity_tokens_list)
		LOGGER.info(f"Running score computation with first {n_ent} entities!!!")
		
		mention_embeds = compute_input_embeddings(
			biencoder=biencoder,
			input_tokens_list=mention_tokens_list,
			batch_size=batch_size
		)
		
		label_embeds = compute_label_embeddings(
			biencoder=biencoder,
			labels_tokens_list=complete_entity_tokens_list,
			batch_size=batch_size
		)

		bienc_ment_to_ent_scores = mention_embeds @ label_embeds.T
		bienc_ment_to_ent_scores = bienc_ment_to_ent_scores.cpu()
		res_file = f"{res_dir}/{data_name}/ment_to_ent_scores_n_m_{n_ment}_n_e_{n_ent}_all_layers_False{misc}.pkl"
		with open(res_file, "wb") as fout:
			dump_dict = {
				"ment_to_ent_scores": bienc_ment_to_ent_scores,
				"mention_tokens_list": mention_tokens_list,
				"args": arg_dict
			}
			pickle.dump(dump_dict, fout)
		
	except Exception as e:
		embed()
		raise e
		
		
					
					
def main():
	data_dir = "../../data/zeshel"
	
	
	worlds = get_zeshel_world_info()
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	parser = argparse.ArgumentParser( description='Run bi-encoder model for computing mention-entity scoring matrix')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
	
	parser.add_argument("--res_dir", type=str, required=True, help="Directory to save results")
	parser.add_argument("--bi_model_file", type=str, default="", help="Biencoder Model file")
	parser.add_argument("--misc", type=str, default="", help="misc suffix for output file")
	
	
	args = parser.parse_args()

	data_name = args.data_name
	n_ment = args.n_ment
	
	batch_size = args.batch_size
	
	res_dir = args.res_dir
	bi_model_file = args.bi_model_file
	misc = args.misc
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
		
	
	iter_worlds = worlds if data_name == "all" else [("", data_name)]
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		run(
			biencoder=biencoder,
			data_info=(data_name, DATASETS[world_name]),
			n_ment=n_ment,
			batch_size=batch_size,
			res_dir=res_dir,
			misc=misc,
			arg_dict=args.__dict__,
		)
	


if __name__ == "__main__":
	main()
