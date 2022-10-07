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

from eval.eval_utils import score_topk_preds, compute_label_embeddings
from eval.run_cross_encoder_w_binenc_retriever_zeshel import _get_cross_enc_pred
from models.crossencoder import CrossEncoderWrapper
from models.params import ENT_END_TAG, ENT_TITLE_TAG, ENT_START_TAG
from utils.data_process import load_entities, load_mentions, get_context_representation
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_MENT_LENGTH, MAX_ENT_LENGTH, MAX_PAIR_LENGTH

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def compute_label_embed_w_e_crossenc(crossencoder, label_tokens_list, batch_size, use_dummy_ment):
	
	try:
		if isinstance(crossencoder, torch.nn.parallel.distributed.DistributedDataParallel):
			crossencoder = crossencoder.module.module
	
		assert isinstance(crossencoder, CrossEncoderWrapper), f"Expected model of type = CrossEncoderWrapper but got of type = {type(crossencoder)}"
		assert not crossencoder.training, "Model should be in eval mode"
		
		if use_dummy_ment:
			# Append some mention-boundary tokens in front of each label tokens
			tokenizer = crossencoder.tokenizer
			dummy_ment = torch.tensor(tokenizer.convert_tokens_to_ids(["[CLS]", ENT_START_TAG, ENT_END_TAG, "[SEP]"]))
			label_tokens_list_w_dummy = [torch.cat((dummy_ment, label_tokens[1:])) for label_tokens in label_tokens_list]
			label_tokens_list_w_dummy = torch.stack(label_tokens_list_w_dummy)
			dataloader = DataLoader(TensorDataset(label_tokens_list_w_dummy), batch_size=batch_size, shuffle=False)
			first_segment_end = len(dummy_ment)
		else:
			dataloader = DataLoader(TensorDataset(label_tokens_list), batch_size=batch_size, shuffle=False)
			first_segment_end = 0
			
		
		with torch.no_grad():
			
			all_encodings = []
			LOGGER.info(f"Starting embedding data with n_data={len(label_tokens_list)}")
			LOGGER.info(f"Bi encoder model device {crossencoder.device}")
			for batch_idx, (batch_data,) in tqdm(enumerate(dataloader), position=0, leave=True, total=len(dataloader)):
				batch_data =  batch_data.to(crossencoder.device)
				encodings = crossencoder.encode_label(
					label_token_idxs=batch_data,
					first_segment_end=first_segment_end
				)
				all_encodings += [encodings.cpu()]
				torch.cuda.empty_cache()
				
			all_encodings = torch.cat(all_encodings)
			
		return all_encodings
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def run(crossencoder, data_fname, n_ment, batch_size, top_k, res_dir, dataset_name, use_dummy_ment, misc, run_exact_reranking_opt, arg_dict):
	try:
		
		assert top_k > 1
	
		crossencoder.eval()
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = crossencoder.tokenizer
		
		test_data = load_mentions(
			mention_file=data_fname["ment_file"],
			kb_id2local_id=kb_id2local_id
		)
		
		test_data = test_data[:n_ment] if n_ment > 0 else test_data
		# First extract all mentions and tokenize them
		mention_tokens_list = [
			get_context_representation(
				sample=mention,
				tokenizer=tokenizer,
				max_seq_length=MAX_MENT_LENGTH)["ids"]
			for mention in tqdm(test_data)
		]
		
		curr_mentions_tensor = torch.LongTensor(mention_tokens_list)
		curr_gt_labels = np.array([x["label_id"] for x in test_data])
		
		dataloader = DataLoader(TensorDataset(curr_mentions_tensor), batch_size=batch_size, shuffle=False)
		
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		
		candidate_encoding  = compute_label_embed_w_e_crossenc(
			crossencoder=crossencoder,
			label_tokens_list=complete_entity_tokens_list,
			batch_size=batch_size,
			use_dummy_ment=use_dummy_ment
		)
		candidate_encoding = candidate_encoding.t() # Take transpose for easier matrix multiplication ops later
		
		assert isinstance(crossencoder, CrossEncoderWrapper)
		init_topk_preds = []
		rerank_topk_preds = []
		with torch.no_grad():
			torch.cuda.empty_cache()
			LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment}, top_k={top_k}")
			LOGGER.info(f"Bi encoder model device {crossencoder.device}")
			for batch_idx, (batch_ment_tokens,) in tqdm(enumerate(dataloader), position=0, leave=True, total=len(dataloader)):

				batch_ment_tokens =  batch_ment_tokens.to(crossencoder.device)

				ment_encodings = crossencoder.encode_input(
					input_token_idxs=batch_ment_tokens,
					first_segment_end=MAX_MENT_LENGTH
				)
				# ment_encodings = biencoder.encode_input(batch_ment_tokens)

				ment_encodings = ment_encodings.to(candidate_encoding.device)
				batch_scores = ment_encodings.mm(candidate_encoding)

				# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
				# If it were complete mention-entity matrix then we would have to use ment_idx
				batch_top_k_scores, batch_top_k_indices = batch_scores.topk(top_k)

				# # Use cross-encoder for re-ranking bi-encoder and randomly sampled entities
				batch_crossenc_topk_scores = _get_cross_enc_pred(
					crossencoder=crossencoder,
					max_pair_length=MAX_PAIR_LENGTH,
					max_ment_length=MAX_MENT_LENGTH,
					batch_ment_tokens=batch_ment_tokens,
					complete_entity_tokens_list=complete_entity_tokens_list,
  					batch_retrieved_indices=batch_top_k_indices,
					use_all_layers=False
				)

				init_topk_preds += [(batch_top_k_indices, batch_top_k_scores)]
				# rerank_topk_preds += [(batch_top_k_indices, batch_top_k_scores)]
				rerank_topk_preds += [(batch_top_k_indices, batch_crossenc_topk_scores)]


		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)

		init_topk_preds = _get_indices_scores(init_topk_preds)
		rerank_topk_preds = _get_indices_scores(rerank_topk_preds)

		json.dump(curr_gt_labels.tolist(), open(f"{curr_res_dir}/gt_labels.txt", "w"))
		json.dump(init_topk_preds, open(f"{curr_res_dir}/e-crossenc_topk_preds.txt", "w"))
		json.dump(rerank_topk_preds, open(f"{curr_res_dir}/e-crossenc_rerank_topk_preds.txt", "w"))

		res = {
			"e-crossenc_init": score_topk_preds(
				gt_labels=curr_gt_labels,
				topk_preds=init_topk_preds
			),
			"e-crossenc_rerank": score_topk_preds(
				gt_labels=curr_gt_labels,
				topk_preds=rerank_topk_preds
			),
			"arg_dict":arg_dict
		}
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))

		########################################## RUN APPROX EXHAUSTIVE INFERERNCE ###################################
		
		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
		if run_exact_reranking_opt:
			run_approx_exhaustive_reranking(
				crossencoder=crossencoder,
				curr_res_dir=curr_res_dir,
				all_mentions_tensor=curr_mentions_tensor,
				label_encoding=candidate_encoding,
				complete_entity_tokens_list=complete_entity_tokens_list
			)
			
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def run_approx_exhaustive_reranking(curr_res_dir, crossencoder, all_mentions_tensor, label_encoding, complete_entity_tokens_list):
	"""
	Runs retrieve and re-rank by with given top_k for all mentions.
	For mentions whose gold entity is not present in top_k, we find rank of gold entity of such mentions,
	retrieve entities up to that using biencoder so that we can retrieve gold entity, and then re-rank entities using
	cross-encoder.
	This is an optimistic upperbound on performance of exhaustive re-ranking as we are avoiding computing cross-encoder
	scores for all entities for a given mention. It might be the case that the top-scoring entity wrt cross-encoder
	is ranked lower than ground-truth entity by the biencoder. In that case, exhaustive inference will incorrectly
	assign each wrong entity to the mention while we will never score any entity beyond the ground-truth entity, so
	we will avoid that sort of mistake in this function
	
	:return:
	"""
	
	try:
		n_ments = len(all_mentions_tensor)
		n_ents = len(complete_entity_tokens_list)
		
		assert label_encoding.shape[1] == n_ents, f"label_encoding.shape[1] = {label_encoding.shape[1]} != n_ents = {n_ents}"
		
		
		# curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
		with open(f"{curr_res_dir}/gt_labels.txt", "r") as fin:
			curr_gt_labels = json.load(fin)
	
		with open(f"{curr_res_dir}/e-crossenc_topk_preds.txt", "r") as fin:
			bienc_topk_preds = json.load(fin)
	
	
	
		# Find indices of mentions for which gold entity is not present in original top_k entities wrt biencoder
		ments_for_exact_inf = []
		for ment_idx, (gt_entity, bienc_topk_ents) in enumerate(zip(curr_gt_labels, bienc_topk_preds["indices"])):
			if gt_entity not in bienc_topk_ents:
				ments_for_exact_inf += [(ment_idx, gt_entity)]
		
		
		# Now run (almost) exhaustive inference for these mentions using cross-encoder model
		all_crossenc_eval_list = []
		for ment_idx, gt_entity in ments_for_exact_inf:
			curr_ment_tokens = all_mentions_tensor[ment_idx].unsqueeze(0).to(crossencoder.device)
			ment_encoding = crossencoder.encode_input(curr_ment_tokens).cpu()
			
			# Compute score for all entities
			ent_scores = ment_encoding.mm(label_encoding)
			
			sorted_ent_scores, sorted_ent_idxs = zip(*sorted(zip(ent_scores.squeeze(0), list(range(n_ents))), reverse=True))
		
			# Find index of gt entity in sorted score list
			gt_ent_idx = sorted_ent_idxs.index(gt_entity)
			
			# Get scores of entities up to and including gt_entity
			ents_to_score_w_ce = sorted_ent_idxs[:gt_ent_idx+1]
			
			
			# Score all entities up to and inclduing gt_entity
			ce_ent_dataloader = DataLoader(TensorDataset(torch.LongTensor(ents_to_score_w_ce)), batch_size=32, shuffle=False)
		
			crossenc_scores_list = []
			for ce_ent_idxs in ce_ent_dataloader:
				# Compute cross-encoder scores for these entities
				curr_crossenc_scores  = _get_cross_enc_pred(
					crossencoder=crossencoder,
					max_pair_length=MAX_PAIR_LENGTH,
					max_ment_length=MAX_MENT_LENGTH,
					batch_ment_tokens=all_mentions_tensor[ment_idx].unsqueeze(0),
					complete_entity_tokens_list=complete_entity_tokens_list,
					batch_retrieved_indices=ce_ent_idxs,
					use_all_layers=False
				).cpu().data.numpy()
				
				crossenc_scores_list += [curr_crossenc_scores]
				
			crossenc_scores = np.concatenate(crossenc_scores_list, axis=1).reshape(-1)
			
			# Compute eval metrics for these scores
			crossenc_scores_eval = score_topk_preds(
				gt_labels=[gt_entity],
				topk_preds=_get_indices_scores([([ents_to_score_w_ce], [crossenc_scores])]),
			)
			all_crossenc_eval_list += [crossenc_scores_eval]
		
		
		
		# Now add results for these mentions for which gt entity was not original part of top-k biencoder entities
		
		with open(f"{curr_res_dir}/res.json", "r") as fin:
			res = json.load(fin)
		
		eval_metrics = res["e-crossenc_rerank"].keys()
		
		if len(all_crossenc_eval_list) > 0:
			all_crossenc_eval = {metric:np.mean([float(crossenc_scores[metric]) for crossenc_scores in all_crossenc_eval_list])
								 for metric in eval_metrics}
			all_crossenc_eval = {key:"{:.2f}".format(val) for key,val in all_crossenc_eval.items()}
		else:
			all_crossenc_eval = {metric:"0" for metric in eval_metrics}
		
		res["e-crossenc_rerank_w_approx_exact_inf_only"] = all_crossenc_eval
		res["e-crossenc_rerank_w_approx_exact_inf"] = {}
		
		n_m_for_exact_inf = len(ments_for_exact_inf)
		for metric in eval_metrics:
			# Get original value for this metric. This has no contributions from mentions whose gt-entity was not part of top-k wrt biencoder
			orig_val = float(res["e-crossenc_rerank"][metric])*n_ments
			
			# This is additional score for mentions whose gt-entity was not part of top-k wrt biencoder by increasing top-k value just so that gt entity is part of top-k bienc entities
			addtional_val = float(all_crossenc_eval[metric])*n_m_for_exact_inf # Get original value for this metric
			
			# Combine orig and additional value to get final score for this metric
			val = (orig_val + addtional_val)/n_ments
			res["e-crossenc_rerank_w_approx_exact_inf"][metric] =  "{:.2f}".format(val)
			# assert val == 100 or metric != "recall", f"val = {val} for metric = recall"
		
			
		with open(f"{curr_res_dir}/res_w_approx_exact_inf.json", "w") as fout:
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
			
		# LOGGER.info("Intentional embed")
		# embed()
		# input("")
		
	except Exception as e:
		embed()
		raise e
	


def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	
	if len(indices) > 0 and torch.is_tensor(indices[0]):
		indices, scores = torch.cat(indices), torch.cat(scores)
		indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	else:
		indices, scores = np.concatenate(indices).tolist(), np.concatenate(scores).tolist()
		
	return {"indices":indices, "scores":scores}




def main():
	worlds = get_zeshel_world_info()
	parser = argparse.ArgumentParser( description='Run e-crossencoder model on given mention data')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	parser.add_argument("--use_dummy_ment", type=int, default=1, choices=[0,1], help="Batch size")
	
	parser.add_argument("--model_file", type=str, required=False, default="", help="Model ckpt file or json file")
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	parser.add_argument("--run_exact_reranking_opt", type=int, default=1, help="1 if to run exact inference for mentions whose gt entity is not part of top-k entities"
																			"0 otherwise")
	worlds = get_zeshel_world_info()
	args = parser.parse_args()

	data_dir = args.data_dir
	data_name = args.data_name
	n_ment = args.n_ment
	top_k = args.top_k
	batch_size = args.batch_size
	use_dummy_ment = bool(args.use_dummy_ment)
	run_exact_reranking_opt = bool(args.run_exact_reranking_opt)
	
	model_file = args.model_file

	res_dir = args.res_dir
	misc = args.misc
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	if model_file.endswith(".json"):
		with open(model_file, "r") as fin:
			config = json.load(fin)
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(model_file)
	
	
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		crossencoder=crossencoder,
		data_fname=DATASETS[data_name],
		n_ment=n_ment,
		top_k=top_k,
		batch_size=batch_size,
		use_dummy_ment=use_dummy_ment,
		dataset_name=data_name,
		res_dir=res_dir,
		misc=misc,
		run_exact_reranking_opt = run_exact_reranking_opt,
		arg_dict=args.__dict__
	)



if __name__ == "__main__":
	main()

