import os
import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import wandb
import numpy as np
from IPython import embed
from pathlib import Path
from collections import defaultdict


import faiss
import pickle
from utils.data_process import load_entities, load_mentions, get_context_representation
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from eval.run_cross_encoder_w_binenc_retriever_zeshel import _get_cross_enc_pred, _get_indices_scores

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def run(crossencoder, data_fname, n_ment_start, n_ment, res_dir, dataset_name, misc, arg_dict,
		cur_k_retvr, num_anchor_ents, anchor_ment_to_ent_file):
	
	
	try:
		crossencoder.eval()
		
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = crossencoder.tokenizer
		
		test_data = load_mentions(mention_file=data_fname["ment_file"], kb_id2local_id=kb_id2local_id)
		test_data = test_data[n_ment_start:n_ment_start+n_ment] if n_ment > 0 else test_data
		
		# First extract all mentions and tokenize them
		all_mentions_tensor = torch.LongTensor([get_context_representation(sample=mention, tokenizer=tokenizer, max_seq_length=MAX_MENT_LENGTH)["ids"]
								for mention in tqdm(test_data)])
		
		complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		n_ents = len(complete_entity_tokens_list)
		
		
		# Load label encodings using anchor_ment_to_ent_file
		rng = np.random.default_rng(0)
		anchor_ent_indices = np.array(sorted(rng.choice(n_ents, size=num_anchor_ents, replace=False)))
		
		with open(anchor_ment_to_ent_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			anchor_ment_to_ent_scores = dump_dict["ment_to_ent_scores"] # shape: (n_anc_ments, n_ents)
			
			intersect_mat = anchor_ment_to_ent_scores[:, anchor_ent_indices] # (n_anc_ments, n_anc_ents)
			U = torch.linalg.pinv(intersect_mat) # shape: (n_anc_ents, n_anc_ments)
			
			label_encodings = U @ anchor_ment_to_ent_scores # shape: n_anc_ents, n_ents)
			label_encodings = label_encodings.T # shape: (n_ents, n_anc_ents)
		
		label_encodings = np.ascontiguousarray(label_encodings.cpu().numpy())
		d = label_encodings.shape[-1]
		LOGGER.info(f"Building index over embeddings of shape {label_encodings.shape}")
		index = faiss.IndexFlatIP(d)
		index.add(label_encodings)
		
		all_ment_encodings = []
		cur_topk_preds = []
		crossenc_topk_preds_w_cur_retrvr = []
		# Retrieve cur_k_retvr labels for each instance and score them with cross-encoder model
		with torch.no_grad():

			# LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment}, top_k={top_k}")
			LOGGER.info(f"Cross encoder model device {crossencoder.device}")
			for ment_idx, curr_ment_tokens in tqdm(enumerate(all_mentions_tensor), position=0, leave=True, total=len(all_mentions_tensor)):
				curr_ment_tokens =  curr_ment_tokens.to(crossencoder.device)

				"""
				 batch_ment_tokens expect shape : (batch_size, ment_seq_len).
				 Since we have just 1 mention of shape (ment_seq_len), we will add one more dim to it
				"""
				_get_cross_enc_pred_w_retrvr = lambda retrv_indices : _get_cross_enc_pred(
					crossencoder=crossencoder,
					max_pair_length=MAX_PAIR_LENGTH,
					max_ment_length=MAX_MENT_LENGTH,
					batch_ment_tokens=curr_ment_tokens.unsqueeze(0),
					complete_entity_tokens_list=complete_entity_tokens_list,
					batch_retrieved_indices=retrv_indices,
					use_all_layers=False
				)
				
				# Compute mention-embedding using score against anchor embeddings
				# anchor_ent_indices.reshape(1,-1) - this needs to be reshape as retrv_indices shape should be (1, num_items)
				ment_encodings = _get_cross_enc_pred_w_retrvr(anchor_ent_indices.reshape(1,-1))
				
				# Find top-indices using approx CUR scores
				cur_top_k_scores, cur_top_k_indices = index.search(ment_encodings.detach().cpu().numpy(), k=cur_k_retvr)

				# Compute cross-encoder scores for labels retrieved using CUR method
				crossenc_topk_scores_w_bienc_retrvr = _get_cross_enc_pred_w_retrvr(cur_top_k_indices)
				crossenc_topk_scores_w_bienc_retrvr = crossenc_topk_scores_w_bienc_retrvr.cpu().data.numpy()
				
				assert ment_encodings.shape[0] == 1
				all_ment_encodings += [ment_encodings[0].cpu().numpy().tolist()]
				cur_topk_preds += [(cur_top_k_indices, cur_top_k_scores)]
				crossenc_topk_preds_w_cur_retrvr += [(cur_top_k_indices, crossenc_topk_scores_w_bienc_retrvr)]

				wandb.log({
					"ment_idx": ment_idx,
					"frac_done": float(ment_idx)/len(all_mentions_tensor)
				})
		
				
		crossenc_topk_preds_w_cur_retrvr = _get_indices_scores(crossenc_topk_preds_w_cur_retrvr)
		
		curr_res_dir = f"{res_dir}/{dataset_name}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		with open(f"{curr_res_dir}/crossenc_w_cur_retrvr_nm={n_ment}_nm_start={n_ment_start}_k_retvr={cur_k_retvr}_{misc}.txt", "w") as fout:
			crossenc_topk_preds_w_cur_retrvr["arg_dict"] = arg_dict
			json.dump(crossenc_topk_preds_w_cur_retrvr, fout)
		
		with open(f"{curr_res_dir}/cur_anchor_ents_nm={n_ment}_nm_start={n_ment_start}_k_retvr={cur_k_retvr}_{misc}.txt", "w") as fout:
			dump_dict = {
				"anchor_ents": anchor_ent_indices.tolist(),
				"ment_embeds": all_ment_encodings,
				"arg_dict": arg_dict
			}
			json.dump(dump_dict, fout)
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e



	



def main():
	
	worlds = get_zeshel_world_info()
	

	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using CUR method')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment_start", type=int, default=0, help="Star offset for mentions")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--cur_k_retvr", type=int, default=500, help="Number of items to retrieve using CUR for re-ranking with exact cross-encoder model")
	parser.add_argument("--num_anchor_ents", type=int, default=500, help="Number of anchor items to use with CUR method")
	
	parser.add_argument("--cross_model_file", type=str, default="", help="Crossencoder Model config file or checkpoint file")
	parser.add_argument("--anchor_ment_to_ent_file", type=str, default="", help="File storing scores between mention anchors and all entities")
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	args = parser.parse_args()

	data_name = args.data_name
	n_ment = args.n_ment
	n_ment_start = args.n_ment_start
	cur_k_retvr = args.cur_k_retvr
	num_anchor_ents = args.num_anchor_ents
	
	cross_model_file = args.cross_model_file
	anchor_ment_to_ent_file = args.anchor_ment_to_ent_file
	
	data_dir = args.data_dir
	res_dir = args.res_dir
	misc = args.misc
	disable_wandb = bool(args.disable_wandb)
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	config = {
			"goal": "Compute entity-entity pairwise similarity matrix",
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(args.__dict__)
	
	try:
		wandb.init(
			project="CrossEnc-HardNeg-Mining-w-CUR",
			dir=res_dir,
			config=config,
			mode="disabled" if disable_wandb else "online"
		)
	except:
		try:
			wandb.init(
				project="CrossEnc-HardNeg-Mining-w-CUR",
				dir=res_dir,
				config=config,
				settings=wandb.Settings(start_method="fork"),
				mode="disabled" if disable_wandb else "online"
			)
		
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(
				project="CrossEnc-HardNeg-Mining-w-CUR",
				dir=res_dir,
				config=config,
				mode="disabled" if disable_wandb else "offline"
			)
	
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		crossencoder=crossencoder,
		data_fname=DATASETS[data_name],
		n_ment_start=n_ment_start,
		n_ment=n_ment,
		cur_k_retvr=cur_k_retvr,
		num_anchor_ents=num_anchor_ents,
		anchor_ment_to_ent_file=anchor_ment_to_ent_file,
		dataset_name=data_name,
		res_dir=res_dir,
		misc=misc,
		arg_dict=args.__dict__,
	)


if __name__ == "__main__":
	main()

