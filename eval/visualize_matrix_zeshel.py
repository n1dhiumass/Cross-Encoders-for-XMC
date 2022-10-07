import os
import sys
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




sys.path.append('../') # To import custom modules
from eval.eval_utils import score_topk_preds


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



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


def compute_ment_embeddings(biencoder, mention_tokens_list):
	with torch.no_grad():
		torch.cuda.empty_cache()
		biencoder.eval()
		bienc_ment_embedding = []
		all_mention_tokens_list_gpu = torch.tensor(mention_tokens_list).to(biencoder.device)
		for ment in tqdm(all_mention_tokens_list_gpu):
			ment = ment.unsqueeze(0)
			bienc_ment_embedding += [biencoder.encode_input(ment)]

		bienc_ment_embedding = torch.cat(bienc_ment_embedding)

	return bienc_ment_embedding


def compute_bienc_ment_to_ent_matrix(biencoder, mention_tokens_list, candidate_encoding):
	bienc_ment_embedding = compute_ment_embeddings(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
	bienc_all_ment_to_ent_scores = bienc_ment_embedding @ candidate_encoding.T
	return bienc_all_ment_to_ent_scores


def visualize_heat_map(val_matrix, title, curr_res_dir):
	"""
	Plot a heat map using give matrix and add x-/y-ticks and title
	:param val_matrix: Matrix for plotting heat map
	:param title: title
	:param curr_res_dir:
	:return:
	"""
	Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
	fig, ax = plt.subplots(figsize=(80,8))

	im = ax.imshow(val_matrix)

	ax.set_title(title)
	ax.set_xlabel("Entities")
	ax.set_ylabel("Mentions")
	fig.tight_layout()
	plt.colorbar(im)
	plt.savefig(f"{curr_res_dir}/{title}.pdf")


def visualize_hist(val_matrix, title, topk, gt_labels, out_filename, plot_individual=False):
	"""
	Plots scores as a histogram{
	:param val_matrix: Matrix for plotting heat map
	:param title: title
	:param curr_res_dir:
	:return:
	"""
	# Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
	
	if isinstance(val_matrix, torch.Tensor):
		val_matrix = val_matrix.cpu().numpy()
	min_score = np.min(val_matrix)
	max_score = np.max(val_matrix)
	bins = np.arange(min_score, max_score, (max_score - min_score)/100)
	
	fig, ax = plt.subplots(figsize=(10,8))
	for i, (row,gt) in tqdm(enumerate(zip(val_matrix, gt_labels)), position=0, leave=True, total=len(val_matrix)):
		_, bins, _ = plt.hist(x=row, bins=bins, color="lightcoral", label="All Scores", alpha=0.1)
		if topk:
			topk_row = row[np.argpartition(row, -1*topk)[-topk:]]
			plt.hist(x=topk_row, bins=bins, color="lightgreen", label=f"Top-{topk}", alpha=0.1)
	
	for i, (row,gt) in tqdm(enumerate(zip(val_matrix, gt_labels)), position=0, leave=True, total=len(val_matrix)):
		plt.hist(x=[row[gt]], bins=bins, color='forestgreen', label="Top Score")
		
	plt.yscale('log')
	ax.set_title(title)
	ax.set_xlabel("Score Range")
	ax.set_ylabel("Number of ment-ent pairs")
	fig.tight_layout()
	out_dir = os.path.dirname(out_filename)
	Path(out_dir).mkdir(exist_ok=True, parents=True)
	plt.savefig(out_filename)
	plt.close()
	
	if not plot_individual:
		return
		
	for i, (row,gt) in tqdm(enumerate(zip(val_matrix, gt_labels)), position=0, leave=True, total=len(val_matrix)):
		plt.clf()
		fig, ax = plt.subplots(figsize=(10,8))
	
		_, bins, _ = plt.hist(x=row, bins=bins, color="lightcoral", label="All Scores")
		
		if topk:
			topk_row = row[np.argpartition(row, -1*topk)[-topk:]]
			plt.hist(x=topk_row, bins=bins, color="lightgreen", label=f"Top-{topk}")
		
		plt.hist(x=[row[gt]], bins=bins, color='forestgreen', label="Top Score")
		plt.yscale('log')
		ax.set_title(title)
		ax.set_xlabel("Score Range")
		ax.set_ylabel("Number of ment-ent pairs")
		fig.tight_layout()
		plt.legend()
		plt.savefig(f"{out_filename}_{i}.pdf")
		plt.close()
  
  
def run(res_dir, data_info, biencoder=None):

	Path(res_dir).mkdir(exist_ok=True, parents=True)
	data_name, data_fnames = data_info

	LOGGER.info("Loading precomputed ment_to_ent scores")
	with open(data_fnames["crossenc_ment_to_ent_scores"], "rb") as fin:
		(ment_to_ent_scores,
		 test_data,
		 mention_tokens_list,
		 entity_id_list,
		 entity_tokens_list) = pickle.load(fin)


	# # Map entity ids to local ids
	ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
	curr_gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)


	####################### RUN MATRIX APPROXIMATION USING CROSSENCODER SCORE MATRIX ###########################
	
	ment_to_ent_scores = ment_to_ent_scores.cpu().numpy()
	# visualize_hist(val_matrix=ment_to_ent_scores,
	#            title=f"{data_name}_crossenc_topk_100",
	#            curr_res_dir=res_dir + "/crossenc_individual_scores",
	#            topk=100,
	#            gt_labels=curr_gt_labels)
	
	norm_ment_to_ent_scores = 1 / (1 + np.exp(-ment_to_ent_scores))
	visualize_hist(val_matrix=norm_ment_to_ent_scores,
			   title=f"{data_name}_norm_crossenc_topk_100",
			   curr_res_dir=res_dir + "/norm_crossenc_individual_scores",
			   topk=100,
			   gt_labels=curr_gt_labels)

	######################## RUN MATRIX APPROXIMATION USING BIENCODER SCORE MATRIX #############################

	if biencoder is not None:
		LOGGER.info("Loading precomputed entity encodings computed using biencoder")
		candidate_encoding = np.load(data_fnames["ent_embed_file"])

		# Keep only encoding of entities for which crossencoder scores are computed i.e. entities in entity_id_list
		candidate_encoding = torch.Tensor(candidate_encoding[entity_id_list])

		mention_tokens_list = torch.LongTensor(mention_tokens_list)
		ment_to_ent_scores = compute_bienc_ment_to_ent_matrix(biencoder=biencoder,
															  mention_tokens_list=mention_tokens_list,
															  candidate_encoding=candidate_encoding)
		
		ment_to_ent_scores = ment_to_ent_scores.cpu().numpy()
		# visualize_hist(val_matrix=ment_to_ent_scores,
		#    title=f"{data_name}_bienc_topk_100",
		#    curr_res_dir=res_dir + "/bienc_individual_scores",
		#    topk=100,
		#    gt_labels=curr_gt_labels)
		
		norm_ment_to_ent_scores = 1 / (1 + np.exp(-ment_to_ent_scores))
		visualize_hist(val_matrix=norm_ment_to_ent_scores,
				   title=f"{data_name}_norm_bienc_topk_100",
				   curr_res_dir=res_dir + "/norm_bienc_individual_scores",
				   topk=100,
				   gt_labels=curr_gt_labels)



			

def main():
	
	###################################### LOAD PRETRAINED MODELS ######################################################
	pretrained_dir = "../../../BLINK_models"
	PARAMETERS = {
			"biencoder_model": f"{pretrained_dir}/biencoder_wiki_large.bin",
			"biencoder_config": f"{pretrained_dir}/biencoder_wiki_large.json",
			"crossencoder_model": f"{pretrained_dir}/crossencoder_wiki_large.bin",
			"crossencoder_config": f"{pretrained_dir}/crossencoder_wiki_large.json",
		}
	model_args = argparse.Namespace(**PARAMETERS)
	(
		biencoder,
		biencoder_params,
	) = load_models(biencoder_model=model_args.biencoder_model, biencoder_config=model_args.biencoder_config)
	
	####################################################################################################################
	
	################################################ DATASET INFO ######################################################
	exp_id = "4_DomainTransfer"
	data_dir = "../../../data/zeshel"
	res_dir = f"../../../results/{exp_id}"
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
	DATASETS["lego"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/lego/ment_to_ent_scores_n_m_100_n_e_10076.pkl"
	DATASETS["star_trek"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/star_trek/ment_to_ent_scores_n_m_100_n_e_34430.pkl"
	DATASETS["forgotten_realms"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/forgotten_realms/ment_to_ent_scores_n_m_100_n_e_15603.pkl"
	DATASETS["yugioh"]["crossenc_ment_to_ent_scores"] =  f"{res_dir}/yugioh/ment_to_ent_scores_n_m_100_n_e_10031.pkl"
	
	####################################################################################################################
	
	world_name = "lego"
	run(res_dir=f"{res_dir}/{world_name}",
		data_info=(world_name, DATASETS[world_name]),
		biencoder=biencoder)


if __name__ == "__main__":
	main()

