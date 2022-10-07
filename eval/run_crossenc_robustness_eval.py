import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from utils.data_process import load_entities, load_mentions
from eval.eval_utils import score_topk_preds, compute_label_embeddings
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from models.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _get_mention_in_context(sample, mention_tokens, tokenizer, max_seq_length):
	
	context_left = sample["context_left"]
	context_right = sample["context_right"]
	context_left = tokenizer.tokenize(context_left)
	context_right = tokenizer.tokenize(context_right)

	left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
	right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
	left_add = len(context_left)
	right_add = len(context_right)
	if left_add <= left_quota:
		if right_add > right_quota:
			right_quota += left_quota - left_add
	else:
		if right_add <= right_quota:
			left_quota += right_quota - right_add

	context_tokens = (
		context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
	)
	
	return context_tokens



def get_mention_rep(sample, tokenizer, max_seq_length, token_opt):
	
	mention_tokens = []
	if sample["mention"] and len(sample["mention"]) > 0:
		mention_tokens = tokenizer.tokenize(sample["mention"])
		if token_opt == "default":
			mention_tokens = [ENT_START_TAG] + mention_tokens + [ENT_END_TAG]
		elif token_opt == "swap_end_start":
			mention_tokens = [ENT_END_TAG] + mention_tokens + [ENT_START_TAG]
			
	ment_in_context_tokens = _get_mention_in_context(
		sample=sample,
		tokenizer=tokenizer,
		mention_tokens=mention_tokens,
		max_seq_length=max_seq_length
	)
	
	rng = np.random.default_rng(0)
	if token_opt == "rand_bndry_w_same_span_len":
		# Randomly insert mention start and end tokens
		ment_start = rng.integers(len(ment_in_context_tokens) - len(mention_tokens))
		ment_end = ment_start + len(mention_tokens)
		
		ment_in_context_tokens = ment_in_context_tokens[:ment_start] \
								 + [ENT_START_TAG] \
								 + ment_in_context_tokens[ment_start:ment_end] \
								 + [ENT_END_TAG] \
								 + ment_in_context_tokens[ment_end:]
	
	
	# Append CLS and SEP tokens
	context_tokens = ["[CLS]"] + ment_in_context_tokens + ["[SEP]"]
	input_ids = tokenizer.convert_tokens_to_ids(context_tokens)[:max_seq_length]
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	assert len(input_ids) == max_seq_length, f"Input_ids len = {len(input_ids)} != max_seq_len ({max_seq_length})"

	return {
		"tokens": context_tokens,
		"ids": input_ids,
	}



def _get_cross_enc_pred(crossencoder, max_ment_length, max_pair_length, batch_ment_tokens, complete_entity_tokens_list, batch_retrieved_indices):
		
	try:
		# Create pair of input,nnbr entity tokens. Strip off first token from nnbr entity as that is CLS token and also limit to max_pair_length
		batch_nnbr_ent_tokens = [complete_entity_tokens_list[nnbr_indices].unsqueeze(0) for nnbr_indices in batch_retrieved_indices.cpu().data.numpy()]
		batch_nnbr_ent_tokens = torch.cat(batch_nnbr_ent_tokens).to(batch_ment_tokens.device)
		
		batch_paired_inputs = []
		for i, ment_tokens in enumerate(batch_ment_tokens):
			paired_inputs = torch.stack([torch.cat((ment_tokens.view(-1), nnbr[1:]))[:max_pair_length] for nnbr in batch_nnbr_ent_tokens[i]])
			# paired_inputs = torch.stack([torch.cat((nnbr[1:], ment_tokens.view(-1)[1:]))[:max_pair_length] for nnbr in batch_nnbr_ent_tokens[i]])
			batch_paired_inputs += [paired_inputs]

		batch_paired_inputs = torch.stack(batch_paired_inputs).to(crossencoder.device)

		batch_crossenc_scores = crossencoder.score_candidate(batch_paired_inputs, first_segment_end=max_ment_length)
		
		batch_crossenc_scores = batch_crossenc_scores.to(batch_retrieved_indices.device)
		
		# Since we just have nearest nbr entities, we need to map argmax in crossenc_scores to original ent id
		batch_crossenc_pred = batch_retrieved_indices.gather(1, torch.argmax(batch_crossenc_scores, dim=1, keepdim=True))
		batch_crossenc_pred = batch_crossenc_pred.view(-1)
		
		return batch_crossenc_pred, batch_crossenc_scores
	except Exception as e:
		embed()
		raise e



def get_tokenized_mentions(tokenizer, test_data, token_opt):
	
	supported_opts = ["default", "rand_bndry_w_same_span_len", "no_bndry", "swap_end_start"]
	assert token_opt in supported_opts, f"Tokenization opt = {token_opt} not supported. Choose one of {supported_opts}"
	tokenized_mentions = torch.LongTensor(
		[
			get_mention_rep(
				sample=mention,
				tokenizer=tokenizer,
				max_seq_length=MAX_MENT_LENGTH,
				token_opt=token_opt
			)["ids"]
			for mention in tqdm(test_data)
		]
	)
	return tokenized_mentions
	
	

def run(biencoder, crossencoder, data_fname, n_ment_arg, batch_size, top_k, res_dir, dataset_name, misc, arg_dict):
	try:
		assert top_k > 1
		biencoder.eval()
		crossencoder.eval()
		
		(title2id, id2title, id2text, kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		test_data = load_mentions(mention_file=data_fname["ment_file"], kb_id2local_id=kb_id2local_id)
		test_data = test_data[:n_ment_arg] if n_ment_arg > 0 else test_data
	
		tokenized_entities = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		label_embeds = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=tokenized_entities, batch_size=batch_size)
		
		token_opt_vals = ["default", "rand_bndry_w_same_span_len", "no_bndry", "swap_end_start"]
		token_opt_vals = ["swap_end_start"]
		for token_opt in token_opt_vals:
			LOGGER.info(f"Running w/ token_opt = {token_opt}")
			run_w_given_token_opt(
				biencoder=biencoder,
				crossencoder=crossencoder,
				test_data=test_data,
				label_embeds=label_embeds,
				tokenized_entities=tokenized_entities,
				batch_size=batch_size,
				n_ment_arg=n_ment_arg,
				top_k=top_k,
				res_dir=res_dir,
				dataset_name=dataset_name,
				misc=misc,
				token_opt=token_opt,
				arg_dict=arg_dict
			)
			
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()


def run_w_given_token_opt(
		biencoder,
		crossencoder,
		test_data,
		label_embeds,
		tokenized_entities,
		batch_size,
		n_ment_arg,
		top_k,
		res_dir,
		dataset_name,
		misc,
		token_opt,
		arg_dict
):
	try:
		
		label_embeds = label_embeds.t() # Take transpose for easier matrix multiplication ops later
		gt_labels = np.array([x["label_id"] for x in test_data])
		
		# First extract all mentions and tokenize them
		tokenized_mentions = get_tokenized_mentions(
			tokenizer=crossencoder.tokenizer,
			test_data=test_data,
			token_opt=token_opt
		)
		
		tokenized_mentions_def_opt = get_tokenized_mentions(
			tokenizer=crossencoder.tokenizer,
			test_data=test_data,
			token_opt="default"
		)
		batched_data = TensorDataset(tokenized_mentions_def_opt, tokenized_mentions, torch.LongTensor(gt_labels))
		bienc_dataloader = DataLoader(batched_data, batch_size=batch_size, shuffle=False)
	
		bienc_topk_preds = []
		crossenc_topk_preds_w_bienc_retrvr = []
		with torch.no_grad():
			
			LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment_arg}, top_k={top_k}")
			LOGGER.info(f"Bi encoder model device {biencoder.device}")
			LOGGER.info(f"Cross encoder model device {crossencoder.device}")
			for batch_idx, (batch_ment_tokens_def_opt, batch_ment_tokens, batch_ment_gt_labels) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
				batch_ment_tokens_def_opt =  batch_ment_tokens_def_opt.to(biencoder.device)
				batch_ment_tokens =  batch_ment_tokens.to(biencoder.device)
				
				ment_encodings = biencoder.encode_input(batch_ment_tokens_def_opt).cpu()
				batch_bienc_scores = ment_encodings.mm(label_embeds)
				
				# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
				# If it were complete mention-entity matrix then we would have to use ment_idx
				batch_bienc_top_k_scores, batch_bienc_top_k_indices = batch_bienc_scores.topk(top_k)
				
				
				_get_cross_enc_pred_w_retrvr = lambda retrv_indices : _get_cross_enc_pred(crossencoder=crossencoder,
																						  max_pair_length=MAX_PAIR_LENGTH,
																						  max_ment_length=MAX_MENT_LENGTH,
																						  batch_ment_tokens=batch_ment_tokens,
																						  complete_entity_tokens_list=tokenized_entities,
																						  batch_retrieved_indices=retrv_indices)
				
				# Use cross-encoder for re-ranking bi-encoder and randomly sampled entities
				batch_crossenc_pred_w_bienc_retrvr, batch_crossenc_topk_scores_w_bienc_retrvr = _get_cross_enc_pred_w_retrvr(batch_bienc_top_k_indices)
				
				bienc_topk_preds += [(batch_bienc_top_k_indices, batch_bienc_top_k_scores)]
				crossenc_topk_preds_w_bienc_retrvr += [(batch_bienc_top_k_indices, batch_crossenc_topk_scores_w_bienc_retrvr)]
				
		
		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment_arg}_k={top_k}_{batch_size}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		bienc_topk_preds = _get_indices_scores(bienc_topk_preds)
		crossenc_topk_preds_w_bienc_retrvr = _get_indices_scores(crossenc_topk_preds_w_bienc_retrvr)

		
		json.dump(gt_labels.tolist(), open(f"{curr_res_dir}/gt_labels.txt", "w"))
		json.dump(bienc_topk_preds, open(f"{curr_res_dir}/bienc_topk_preds_{token_opt}.txt", "w"))
		json.dump(crossenc_topk_preds_w_bienc_retrvr, open(f"{curr_res_dir}/crossenc_topk_preds_w_bienc_retrvr_{token_opt}.txt", "w"))

		
		with open(f"{curr_res_dir}/res_{token_opt}.json", "w") as fout:
			res = {"bienc": score_topk_preds(gt_labels=gt_labels,
											 topk_preds=bienc_topk_preds),
				   "crossenc_w_bienc_retrvr": score_topk_preds(gt_labels=gt_labels,
															   topk_preds=crossenc_topk_preds_w_bienc_retrvr),
				   "extra_info": {"arg_dict": arg_dict}}
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()




def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	return {"indices":indices, "scores":scores}


def main():
	
	worlds = get_zeshel_world_info()
	

	parser = argparse.ArgumentParser( description='Run cross-encoder model after perturbing mention boundary tokens')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	
	
	parser.add_argument("--bi_model_file", type=str, required=True, help="Biencoder Model config file or checkpoint file")
	parser.add_argument("--cross_model_file", type=str, default="", help="Crossencoder Model config file or checkpoint file")
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	
	args = parser.parse_args()

	data_name = args.data_name
	n_ment = args.n_ment
	top_k = args.top_k
	batch_size = args.batch_size
	
	bi_model_file = args.bi_model_file
	cross_model_file = args.cross_model_file
	
	data_dir = args.data_dir
	res_dir = args.res_dir
	misc = args.misc
	arg_dict = args.__dict__
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	if bi_model_file.endswith(".json"):
		with open(bi_model_file, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
	else:
		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	
	if cross_model_file.endswith(".json"):
		with open(cross_model_file, "r") as fin:
			config = json.load(fin)
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
	
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	
	
	
	LOGGER.info(f"Running inference for world = {data_name}")
	
	run(
		biencoder=biencoder,
		crossencoder=crossencoder,
		data_fname=DATASETS[data_name],
		n_ment_arg=n_ment,
		top_k=top_k,
		batch_size=batch_size,
		dataset_name=data_name,
		res_dir=res_dir,
		misc=misc,
		arg_dict=arg_dict
	)
		


if __name__ == "__main__":
	main()

