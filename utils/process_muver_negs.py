import os
import sys
import json
import torch
import types
import wandb
import pickle
import warnings
import logging
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from IPython import embed
from pathlib import Path
from collections import defaultdict


from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, N_MENTS_ZESHEL
from utils.data_process import load_entities, load_mentions, load_mentions_by_datasplit
from eval.eval_utils import score_topk_preds

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def run(muver_filename, combined_ment_file, zeshel_data_files, res_dir, split_name):
	
	try:
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		
		# Load topk entity data saved by MUVER code
		ent_titles_for_ments = []
		with open(muver_filename, "r") as fin:
			for line in fin:
				ent_titles_for_ments += [json.loads(line)]
		
		
		# Load entities for each domain
		all_entities = {domain: load_entities(entity_file=zeshel_data_files[domain]["ent_file"]) for domain in zeshel_data_files}
		
		# Load mentions for each domain
		all_mentions_per_domain = {
			domain: load_mentions(
				mention_file=zeshel_data_files[domain]["ment_file"],
				kb_id2local_id=all_entities[domain][3]
			)
			for domain in zeshel_data_files
		}
		
		# Load all mentions in this data split using combined_ment_file
		all_mentions = load_mentions_by_datasplit(
			mention_file=combined_ment_file,
			all_kb_id2local_id={domain:all_entities[domain][3] for domain in all_entities}
		)
		
		# Map storing title to id mapping for each entity in each domain
		title2id = {domain:all_entities[domain][0] for domain in all_entities}
		
		
		# Iterate over all mentions in current data split and put together their topk ents per domain
		topk = 200
		topk_ent_ids_per_domain = defaultdict(list)
		gt_ent_ids_per_domain = defaultdict(list)
		for curr_idx, (curr_ment, curr_ents) in enumerate(zip(all_mentions, ent_titles_for_ments)):
			curr_domain = curr_ment["domain"]
			
			curr_gt_label_id = curr_ment["label_id"]
			curr_ent_ids = [title2id[curr_domain][ent['title']] for ent in curr_ents]
			
			gt_ent_ids_per_domain[curr_domain] += [curr_gt_label_id]
			topk_ent_ids_per_domain[curr_domain] += [curr_ent_ids]
			
			assert len(curr_ent_ids) == topk, f"Number of entities = {len(curr_ent_ids)} != topk = {topk}"
			
		# Make sure that ordering of mentions in separate files containing mentions for each domain is
		# same as the order in which these mention occurred in global file containing mentions of all domains in this split
		# We achieve this by matching gt ids for mentions
		for domain, gt_ent_ids in gt_ent_ids_per_domain.items():
			gt_ent_ids_wrt_domain_ment_file = [ment["label_id"] for ment in all_mentions_per_domain[domain]]
			for gt_ent_id_1, gt_ent_id_2 in zip(gt_ent_ids, gt_ent_ids_wrt_domain_ment_file):
				assert gt_ent_id_1 == gt_ent_id_2
			
	
		# Dump down topk entities for each domain to a file
		domain_scores = {}
		for domain in topk_ent_ids_per_domain:
			res_file = f"{res_dir}/{domain}_topk_ents.json"
			with open(res_file, "w") as fout:
				n_ments = len(topk_ent_ids_per_domain[domain])
				# Assign dummy scores so that even if we try to sort based in decreasing order based on these scores,
				# we do not change the entity ordering
				scores = np.array([np.arange(topk, 0, -1) for _ in range(n_ments)])
				indices = np.array(topk_ent_ids_per_domain[domain])
				
				assert scores.shape == indices.shape, f"scores.shape = {scores.shape} != indices.shape = {indices.shape}"
				
				scores = scores.tolist()
				indices = indices.tolist()
				all_retr_ents_per_domain = {"indices":indices, "scores":scores}
				json.dump(all_retr_ents_per_domain, fout)
				
				
				domain_scores[domain] = score_topk_preds(
					gt_labels=gt_ent_ids_per_domain[domain],
					topk_preds=all_retr_ents_per_domain
				)
				LOGGER.info(f"Domain = {domain}, score = {domain_scores[domain]}")
		
		# Also right down some eval metrics for each domain
		metrics = {m for domain in domain_scores for m in domain_scores[domain]}
		domains = list(domain_scores.keys())
		domain_scores["macro_all"] = {}
		domain_scores["micro_all"] = {}
		
		TOTAL_MENTS = np.sum([N_MENTS_ZESHEL[domain] for domain in domains])
		for curr_met in metrics:
			macro_avg = np.mean([float(domain_scores[domain][curr_met]) for domain in domains])
			domain_scores["macro_all"][curr_met] = "{:.2f}".format(macro_avg)
			
			
			micro_avg = np.sum([N_MENTS_ZESHEL[domain]*float(domain_scores[domain][curr_met])/TOTAL_MENTS for domain in domains])
			domain_scores["micro_all"][curr_met] = "{:.2f}".format(micro_avg)
			
		with open(f"{res_dir}/{split_name}_domain_scores.json", "w") as fout:
			json.dump(domain_scores, fout, indent=4)
		
			
				
		# LOGGER.info("Intentional embed")
		# embed()
	except Exception as e:
		embed()
		raise e


def main():
	data_dir = "../../data/zeshel"
	muver_data = "../../data/muver"
	
	worlds = get_zeshel_world_info()
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=None, worlds=worlds)
	
	split_name_vals = ["train", "test","valid"]
	view_subdir_vals = ["topk_ents_wo_view_merge", "topk_ents_w_view_merge"]
	
	for split_name, view_subdir in itertools.product(split_name_vals, view_subdir_vals):
		run(muver_filename=f"{muver_data}/{view_subdir}/{split_name}_candidates.json",
			combined_ment_file=f"{muver_data}/processed/{split_name}.jsonl",
			zeshel_data_files=DATASETS,
			res_dir=f"{muver_data}/{view_subdir}",
			split_name=split_name)


if __name__ == "__main__":
	main()

