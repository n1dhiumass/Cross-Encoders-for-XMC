import sys
import json
import pickle
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
from os.path import dirname
from itertools import cycle
import matplotlib.pyplot as plt

from eval.eval_utils import compute_overlap
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info

from utils.data_process import load_entities, load_mentions, get_context_representation

cmap = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

from models.crossencoder import to_cross_bert_input
from utils.zeshel_utils import MAX_MENT_LENGTH, MAX_PAIR_LENGTH

max_ment_length = MAX_MENT_LENGTH
max_pair_length = MAX_PAIR_LENGTH


def explore_crossencoder_helper(crossencoder, input_tokens, label_tokens):
	
	try:
		linear_layer = crossencoder.model.encoder.additional_linear
		# crossencoder.model.encoder.bert_model.encoder.output_attentions = True
		# crossencoder.model.encoder.bert_model.encoder.output_hidden_states = True
		
		input_pair_idxs = torch.cat([input_tokens, label_tokens[1:]])[:max_pair_length]
		input_pair_idxs = input_pair_idxs.unsqueeze(0).to(crossencoder.device)
		
		# input_idxs.shape : 1 x max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=crossencoder.tokenizer.pad_token_id, first_segment_end=max_ment_length,
		)
		
		# (crossenc_score, pair_embed, final_token_embs, pair_embed_w_internal_lin, all_hidden_units, all_attention_weights) = crossencoder.model.encoder.forward_per_layer(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size,)
		# pair_embed_per_layer = torch.stack([curr_hidden_units[:, 0, :].squeeze(0) for curr_hidden_units in all_hidden_units])
		# pair_score_per_layer = linear_layer(pair_embed_per_layer)
		
		batch_input = torch.stack([input_pair_idxs, input_pair_idxs, input_pair_idxs, input_pair_idxs, input_pair_idxs, input_pair_idxs])
		
		batch_input = batch_input.view(2,3,-1)
		LOGGER.info(f"Batch input shape = {batch_input.shape}")
		batch_pair_score_per_layer = crossencoder.score_candidate_per_layer(input_pair_idxs=batch_input, first_segment_end=max_ment_length)
		
		
		batch_input = torch.stack([input_pair_idxs, input_pair_idxs, input_pair_idxs, input_pair_idxs, input_pair_idxs])
		LOGGER.info(f"Batch input shape = {batch_input.shape}")
		batch_pair_score_per_layer = crossencoder.score_candidate_per_layer(input_pair_idxs=batch_input, first_segment_end=max_ment_length)
		
		pair_score_per_layer = crossencoder.model.encoder.forward_per_layer(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size,)
		pair_score_per_layer2 = crossencoder.score_candidate_per_layer(input_pair_idxs=input_pair_idxs, first_segment_end=max_ment_length)
		
		
		embed()
		return pair_score_per_layer.detach().cpu().numpy().tolist()
		# output_bert, output_pooler, all_hidden_units, all_attention_weights  = crossencoder.model.encoder.bert_model(input_pair_idxs, segment_idxs, mask)
	except Exception as e:
		embed()
		raise e
	

def explore_crossencoder(crossencoder, mention_tokens_list, curr_gt_labels, complete_entity_tokens_list):
	try:
		crossencoder.eval()
		
		if not torch.is_tensor(mention_tokens_list):
			mention_tokens_list = torch.LongTensor(mention_tokens_list)
		
		for idx,mention_tokens in enumerate(mention_tokens_list[:2]):
			scores0 = explore_crossencoder_helper(crossencoder=crossencoder, input_tokens=mention_tokens,
										label_tokens=complete_entity_tokens_list[0])
			scoresgt = explore_crossencoder_helper(crossencoder=crossencoder, input_tokens=mention_tokens,
										label_tokens=complete_entity_tokens_list[curr_gt_labels[idx]])
			
			LOGGER.info(f"Scores 0 vs gt:\n{np.array(list(zip(scores0, scoresgt)))}\n")
			# LOGGER.info(f"Scores gt:\n{scoresgt}\n")
			
		LOGGER.info("Inside explore crossencoder function")
		embed()
		return {}
	except Exception as e:
		embed()
		raise e


def run(crossencoder, res_dir, data_info, misc):
	try:
		max_ment_length = MAX_MENT_LENGTH
		n_ment = 10
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		data_name, data_fname = data_info
		
		LOGGER.info("Loading precomputed entity encodings computed using biencoder")
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		
		LOGGER.info("Loading all entities")
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		n_ent = len(complete_entity_tokens_list)
		
		LOGGER.info("Loading test samples")
		test_data = load_mentions(mention_file=data_fname["ment_file"],
								  kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[:n_ment] if n_ment > 0 else test_data
		
		tokenizer = crossencoder.tokenizer
		mention_tokens_list = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=max_ment_length)["ids"]
								for mention in tqdm(test_data)]
		
		curr_gt_labels = [mention["label_id"] for mention in test_data]
		
		res = explore_crossencoder(crossencoder=crossencoder,
								   mention_tokens_list=mention_tokens_list,
								   complete_entity_tokens_list=complete_entity_tokens_list,
								   curr_gt_labels=curr_gt_labels)

		temp_res_dir = f"{res_dir}/crossenc_explore"
		Path(temp_res_dir).mkdir(exist_ok=True, parents=True)
		with open(f"{temp_res_dir}/res_{misc}.json", "w") as fout:
			res["dataname"] = data_name
			res["n_ment"] = n_ment
			res["n_ent"] = n_ent
			res["crossenc_score_mat"] = data_fname["crossenc_ment_to_ent_scores"]
			json.dump(obj=res, fp=fout)
		
		return res
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()


def analyse_per_layer_scores(data_info, res_dir, misc):
	
	try:
		
		data_name, data_fname = data_info
		score_file = data_fname["crossenc_ment_to_ent_scores"]
		with open(score_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			ment_to_ent_scores_per_layer = dump_dict["ment_to_ent_scores"]
			# test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			n_layers, n_ment, n_ent = ment_to_ent_scores_per_layer.shape
		
		topk_vals = [10, 100, 1000, 5000]
		# topk_vals = [2,3,4]
		res = {}
		for topk in topk_vals:
			final_topk_scores, final_topk_ents = ment_to_ent_scores_per_layer[-1].topk(topk)
			final_topk_ents = final_topk_ents.numpy()
			for layer_idx in range(n_layers):
				
				curr_topk_scores, curr_topk_ents = ment_to_ent_scores_per_layer[layer_idx].topk(topk)
				overlap = compute_overlap(indices_list1=final_topk_ents, indices_list2=curr_topk_ents.numpy())

				res[f"l={layer_idx}_k={topk}_overlap"] = overlap
			
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		# Also eval each layer for accuracy wrt ground-truth - for sanity check
		out_file = f"{res_dir}/layerwise_recall_{misc}.json"
		with open(out_file, "w") as fout:
			res["data_info"] = data_info
			res["n_ment"] = n_ment
			res["n_ent"] = n_ent
			res["n_layers"] = n_layers
			res["score_file"] = score_file
			res["topk_vals"] = topk_vals
			json.dump(obj=res, fp=fout)
		
		plot_per_layer_recall(res_file=out_file)
	except Exception as e:
		embed()
		raise e
	

def plot_per_layer_recall(res_file):
	
	with open(res_file, "r") as fin:
		res = json.load(fin)

	topk_vals = res["topk_vals"]
	n_layers = res["n_layers"]
	data_info = res["data_info"]
	n_ment = res["n_ment"]
	n_ent = res["n_ent"]
	
	plt.clf()
	for topk, color in zip(topk_vals, cycle(cmap)):
		
		X = np.arange(n_layers)
		Y = [res[f"l={l}_k={topk}_overlap"]["common_frac"][0] for l in X]
		Y = np.array([float(y[5:]) for y in Y])
		Y_err = [res[f"l={l}_k={topk}_overlap"]["common_frac"][1] for l in X]
		Y_err = np.array([float(y[4:]) for y in Y_err])
		
		plt.scatter(X, Y, marker="x", color=color[0], label=f"k={topk}", alpha=0.5)
		plt.scatter(X, Y, color=color[1],s=500*Y_err, alpha=0.5)
		# plt.plot(X, Y, color=color[0], label=f"top-k={topk}", alpha=0.5)
		# plt.fill_between(X, Y-Y_err, Y+Y_err, color=color[1], alpha=0.5)
	
	plt.xlabel("Layer")
	plt.ylabel("Mean recall wrt final layer")
	plt.title(f"Data = {data_info[0]} with {n_ment} mentions and {n_ent} entities")
	plt.legend()
	plt.grid()
	res_dir = f"{dirname(res_file)}/plots"
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	plt.savefig(f"{res_dir}/recall.pdf")
	plt.close()

def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()
	parser = argparse.ArgumentParser( description='Explore functioning of cross-encoder model')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	
	parser.add_argument("--cross_model_config", type=str, required=True, help="Crossencoder Model config file")
	parser.add_argument("--score_file", type=str, required=True, help="pkl file with precomputed crossencoder scores")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix for result file")
	
	args = parser.parse_args()
	data_name = args.data_name

	cross_model_config = args.cross_model_config
	score_file = args.score_file
	misc = "_" + args.misc if args.misc != "" else ""

	
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=None, worlds=worlds)
	DATASETS[data_name]["crossenc_ment_to_ent_scores"] = score_file
	
	
	# res_dir = dirname(dirname(dirname(cross_model_config)))
	# with open(cross_model_config, "r") as fin:
	# 	config = json.load(fin)
	# 	config["bert_args"] = {"output_attentions":True, "output_hidden_states":True}
	# 	crossencoder = CrossEncoderWrapper.load_model(config=config)
	
	res_dir = dirname(score_file)
	iter_worlds = worlds[:4] if data_name == "all" else [("dummy", data_name)]
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		# run(crossencoder=crossencoder, data_info=(world_name, DATASETS[world_name]),
		# 	res_dir=f"{res_dir}/{world_name}", misc=misc)
		
		analyse_per_layer_scores(data_info=(world_name, DATASETS[world_name]),
								 res_dir=f"{res_dir}/layer_analysis", misc=misc)
	

if __name__ == "__main__":
	main()
