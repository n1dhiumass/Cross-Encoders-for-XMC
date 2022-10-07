import os
import sys
import json

import torch
import pickle
import captum
import logging
import argparse
import numpy as np


from pathlib import Path
from tqdm import tqdm

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from pytorch_transformers.modeling_bert import BertModel
from models.crossencoder import CrossEncoderWrapper, CrossEncoderModule, CrossBertWrapper
from utils.data_process import create_input_label_pair
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_MENT_LENGTH
from IPython import embed

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

# nlp = spacy.load('en')




# def interpret_sentence(model, lig, token_reference, sentence, seq_len, gt_label):
#
#
# 	text_token_ids  = model.tokenizer.encode(sentence.lower())
# 	# ans = model.tokenizer.decode(text_token_ids)
# 	# ans2 = model.tokenizer.convert_ids_to_tokens(text_token_ids)
#
# 	if len(text_token_ids) < seq_len: # Pad up to required seq len
# 		text_token_ids += [model.NULL_IDX] * (seq_len - len(text_token_ids))
#
#
# 	return interpret_tokenized_sentence(
# 		model=model,
# 		lig=lig,
# 		token_reference=token_reference,
# 		input_token_ids=text_token_ids,
# 		gt_label=gt_label
# 	)


def interpret_tokenized_sentence(model, lig, token_reference, input_token_ids, gt_label, internal_batch_size=1):
	
	try:
		assert isinstance(lig, LayerIntegratedGradients)
		
		model.zero_grad()
		if not torch.is_tensor(input_token_ids):
			input_indices = torch.tensor(input_token_ids, device=model.device).unsqueeze(0)
		else:
			input_indices = input_token_ids.to(model.device).unsqueeze(0)
			
		seq_len = input_indices.shape[-1]
		
		pred = model.score_candidate(input_pair_idxs=input_indices, first_segment_end=MAX_MENT_LENGTH)
		pred = pred.detach().cpu().numpy()[0]
		pred_label = 0 # Filling in dummy value for this variable as we don't have predicted label as such per input pair
		
		# generate reference indices for each sample
		reference_indices = token_reference.generate_reference(sequence_length=seq_len, device=model.device).unsqueeze(0)
	
		# compute attributions and approximation delta using layer integrated gradients
		attributions_ig, delta = lig.attribute(
			input_indices,
			reference_indices,
			n_steps=500,
			return_convergence_delta=True,
			internal_batch_size=internal_batch_size
			
		)
		
		text_tokens = model.tokenizer.convert_ids_to_tokens(input_token_ids.numpy())
		vis = add_attributions_to_visualizer(
			attributions=attributions_ig,
			text_tokens=text_tokens,
			pred_score=pred,
			pred_label=pred_label,
			gt_label=gt_label,
			delta=delta,
		)
	
		return vis

	except Exception as e:
		LOGGER.info("Error in interpret_sentence")
		embed()
		raise e

	
def add_attributions_to_visualizer(attributions, text_tokens, pred_score, pred_label, gt_label, delta):
	attributions = attributions.sum(dim=2).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	attributions = attributions.cpu().detach().numpy()

	return visualization.VisualizationDataRecord(
		word_attributions=attributions,
		pred_prob=pred_score,
		pred_class=str(pred_label),
		true_class=str(gt_label),
		attr_class=str(1),
		attr_score=attributions.sum(),
		raw_input_ids=text_tokens,
		convergence_score=delta
	)
	

def run_model_interpret_given_mention(model, tokenized_mention, tokenized_entities, entity_scores, k, gt_label, internal_batch_size):
	
	# For each mention,
	# 1) visualize top-k entities that get highest score wrt cross-encoder
	# 2) visualize top-k entities that get highest score wrt bi-encoder
	# 3) visualize random entities
	
	lig = LayerIntegratedGradients(model, model.model.encoder.bert_model.embeddings)
	token_reference = TokenReferenceBase(reference_token_idx=model.NULL_IDX) # Use pad token for generating reference
	
	topk_scores, topk_ents = torch.topk(entity_scores, k=k)
	
	lowestk_scores, lowestk_ents = torch.topk(-1 * entity_scores, k=k)
	
	topk_vis = []
	lowestk_vis = []
	for ent_iter, ent_id in enumerate(topk_ents):
		
		# Visualize mention-entity pair
		ment_ent_pair = create_input_label_pair(
			input_token_idxs=tokenized_mention,
			label_token_idxs=tokenized_entities[ent_id]
		)
		
		topk_vis += [interpret_tokenized_sentence(
			model=model,
			lig=lig,
			token_reference=token_reference,
			input_token_ids=ment_ent_pair,
			gt_label=f"{ent_iter}/id_{ent_id}/gt_{gt_label}",
			internal_batch_size=internal_batch_size
		)]
	
	for ent_iter, ent_id in enumerate(lowestk_ents):
		
		# Visualize mention-entity pair
		ment_ent_pair = create_input_label_pair(
			input_token_idxs=tokenized_mention,
			label_token_idxs=tokenized_entities[ent_id]
		)
		
		lowestk_vis += [interpret_tokenized_sentence(
			model=model,
			lig=lig,
			token_reference=token_reference,
			input_token_ids=ment_ent_pair,
			gt_label=f"{-1*ent_iter}/id_{ent_id}/gt_{gt_label}",
			internal_batch_size=internal_batch_size
		)]
		
		
	
	topk_vis_html = visualization.visualize_text(topk_vis)
	lowestk_vis_html = visualization.visualize_text(lowestk_vis)
	
	
	return {"topk_vis_html": topk_vis_html, "lowestk_vis_html":lowestk_vis_html}


def run_model_interpret(model, res_dir, data_info, k, misc, internal_batch_size):
	
	try:
		# assert isinstance(model, CrossEncoderWrapper)
		# assert isinstance(model.model, CrossEncoderModule)
		# assert isinstance(model.model.encoder, CrossBertWrapper)
		# assert isinstance(model.model.encoder.bert_model, BertModel)
		
		data_name, data_fname = data_info
		model.eval()
		
		# embeddings = model.model.encoder.bert_model.embeddings
		# lig = LayerIntegratedGradients(model, embeddings)
		# token_reference = TokenReferenceBase(reference_token_idx=model.NULL_IDX) # Use pad token for generating reference
		
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
	
		n_ments, n_ents = crossenc_ment_to_ent_scores.shape
	
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
		mentions = [" ".join([ment_dict["context_left"],ment_dict["mention"], ment_dict["context_right"]]) for ment_dict in test_data]
		tokenized_mentions = torch.LongTensor(mention_tokens_list)
		
		complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		tokenized_entities = complete_entity_tokens_list[entity_id_list]
		
		
		for ment_idx in tqdm(range(n_ments)):
			
			ans = run_model_interpret_given_mention(
				model=model,
				tokenized_mention=tokenized_mentions[ment_idx],
				tokenized_entities=tokenized_entities,
				entity_scores=crossenc_ment_to_ent_scores[ment_idx],
				k=k,
				gt_label=gt_labels[ment_idx],
				internal_batch_size=internal_batch_size
			)
			
			for vis_type, vis in ans.items():
				out_file = f"{res_dir}/m={ment_idx}_k={k}_{vis_type}{misc}.html"
				Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
				with open(out_file, "w") as fout:
					fout.write(vis.data)
			
			
	
		
	except Exception as e:
		LOGGER.info(f"Error in run_model_interpret {e}")
		embed()
		raise e




# def run_model_interpret_dummy(model, res_dir):
#
# 	try:
# 		# assert isinstance(model, CrossEncoderWrapper)
# 		# assert isinstance(model.model, CrossEncoderModule)
# 		# assert isinstance(model.model.encoder, CrossBertWrapper)
# 		# assert isinstance(model.model.encoder.bert_model, BertModel)
#
# 		embeddings = model.model.encoder.bert_model.embeddings
# 		lig = LayerIntegratedGradients(model, embeddings)
# 		token_reference = TokenReferenceBase(reference_token_idx=model.NULL_IDX) # Use pad token for generating reference
#
#
# 		vis_data_records_ig = []
# 		sentences = ['It was a fantastic performance !', 'Best film ever', 'It was a horrible movie']
# 		for sentence in sentences:
# 			vis_data_records_ig  += [
# 				interpret_sentence(
# 					model=model,
# 					lig=lig,
# 					token_reference=token_reference,
# 					sentence=sentence,
# 					gt_label=1,
# 					seq_len=7
# 				)
# 			]
#
# 		print('Visualize attributions based on Integrated Gradients')
# 		final_vis = visualization.visualize_text(vis_data_records_ig)
#
# 		out_file = f"{res_dir}/interpret_vis.html"
# 		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
# 		with open(out_file, "w") as fout:
# 			fout.write(final_vis.data)
#
#
#
# 	except Exception as e:
# 		LOGGER.info("Error in main")
# 		embed()
# 		raise e

	
def main():
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run Captum model for interpreting model predictions')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	
	# parser.add_argument("--bi_model_file", type=str, default="", help="Bi-encoder Model config file or ckpt file")
	parser.add_argument("--cross_model_file", type=str, required=True, help="Cross-encoder Model config file or ckpt file")
	parser.add_argument("--k", type=int, default=10, help="k for finding topk entities")
	parser.add_argument("--internal_batch_size", type=int, default=64, help="k for finding topk entities")
	parser.add_argument("--res_dir", type=str, required=True, help="Result dir to store results")
	parser.add_argument("--misc", type=str, default="", help="misc suffix to add to result file")
	
	args = parser.parse_args()
	data_name = args.data_name
	
	# bi_model_file = args.bi_model_file
	cross_model_file = args.cross_model_file
	
	k = args.k
	internal_batch_size = args.internal_batch_size
	res_dir = args.res_dir
	misc = "_" + args.misc if args.misc != "" else ""
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	# cross_model_file = "../.	./results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/model/0-last.ckpt"
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
	
	if cross_model_file.endswith(".json"):
		with open(cross_model_file, "r") as fin:
			config = json.load(fin)
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
	
	crossencoder = crossencoder.cuda()
	
	res_dir = f"{res_dir}/{data_name}"
	run_model_interpret(
		model=crossencoder,
		res_dir=res_dir,
		data_info=(data_name, DATASETS[data_name]),
		k=k,
		internal_batch_size=internal_batch_size,
		misc=misc
	)


if __name__ == "__main__":
	main()