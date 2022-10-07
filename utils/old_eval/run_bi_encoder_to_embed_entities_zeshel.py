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



logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)




def load_models(args):

	# load biencoder model
	with open(args.biencoder_config) as json_file:
		biencoder_params = json.load(json_file)
		biencoder_params["path_to_model"] = args.biencoder_model
	biencoder = load_biencoder(biencoder_params)
	
	return (
		biencoder,
		biencoder_params,
	)


def run(biencoder, dataset_fname, batch_size):
	try:
		# loading tokenized entities
		complete_entity_tokens_list = torch.LongTensor(np.load(dataset_fname["ent_tokens_file"]))
		
		batched_data = TensorDataset(complete_entity_tokens_list)
		bienc_dataloader = DataLoader(batched_data, batch_size=batch_size, shuffle=False)
		
		with torch.no_grad():
			biencoder.eval()
			
			all_ent_encodings = []
			LOGGER.info(f"Starting embedding entities computation with n_ent={len(complete_entity_tokens_list)}")
			LOGGER.info(f"Bi encoder model device {biencoder.device}")
			for batch_idx, (batch_entities,) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
				batch_entities =  batch_entities.to(biencoder.device)
				
				ent_encodings = biencoder.encode_candidate(batch_entities)
				
				all_ent_encodings += [ent_encodings.cpu().numpy()]
				
				
			all_ent_encodings = np.concatenate(all_ent_encodings)
			
			out_file = dataset_fname["ent_embed_file"]
			np.save(file=out_file, arr=all_ent_encodings)
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		time.sleep(2)


def main():
	parser = argparse.ArgumentParser( description='Compute entity encoding using a pretrained biencoder model')
	
	worlds =  ["american_football", "coronation_street", "doctor_who", "elder_scrolls", "fallout", "final_fantasy",
				   "forgotten_realms", "ice_hockey", "lego", "military", "muppets", "pro_wrestling", "star_trek",
				   "starwars", "world_of_warcraft", "yugioh"]
		
		
		
		
	parser.add_argument("--data_name", type=str, choices=worlds + ["all"], help="Dataset name")
	parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
	args = parser.parse_args()
	
	data_name  = args.data_name
	batch_size = args.batch_size
	
	pretrained_dir = "../../BLINK_models"
	data_dir = "../../data/zeshel"
	DATASETS = {world: {"ment_file": f"{data_dir}/processed/train_worlds/{world}_mentions.jsonl",
						"ent_file":f"{data_dir}/documents/{world}.json",
						"ent_tokens_file":f"{data_dir}/tokenized_entities/{world}_128_bert_base_uncased.npy",
						"ent_embed_file":f"{data_dir}/tokenized_entities/{world}_128_bert_base_uncased_embeds.npy"
						}
						for world in worlds
					}
	
	PARAMETERS = {
			"biencoder_model": f"{pretrained_dir}/biencoder_wiki_large.bin",
			"biencoder_config": f"{pretrained_dir}/biencoder_wiki_large.json",
		}
	model_args = argparse.Namespace(**PARAMETERS)
	(
		biencoder,
		biencoder_params,
	) = load_models(model_args)
	
	
	if data_name == "all":
		for world in tqdm(worlds):
			LOGGER.info(f"Generating embedding for world = {world}")
			dataset_fname = DATASETS[world]
			run(biencoder=biencoder, dataset_fname=dataset_fname, batch_size=batch_size)
	else:
		dataset_fname = DATASETS[data_name]
		run(biencoder=biencoder, dataset_fname=dataset_fname, batch_size=batch_size)
	
	

if __name__ == "__main__":
	main()
