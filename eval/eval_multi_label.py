import sys
import json
import logging
import argparse
import numpy as np
from models.biencoder import BiEncoderWrapper
from tqdm import tqdm
from pathlib import Path
from IPython import embed

import torch
import faiss
from utils.data_process import XMCDataset
from utils.data_process import read_xmc_data, tokenize_input
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def process_label_data(
	labels,
	tokenizer,
	max_label_length,
):
	"""
	
	:param input_data: List of raw input datapoints. Each datapoint is dict with keys `input` and `label_idxs`
	:param labels: List of raw labels.
	:param tokenizer:
	:param max_input_length:
	:param max_label_length:
	
	# TODO: Fix docstring
	:return: Dict mapping to list of input tokens, corresponding label tokens etc
	"""
	try:
		processed_labels = []
		for idx, label in enumerate(tqdm(labels)):
			label_tokens = tokenize_input(input=label,
										  tokenizer=tokenizer,
										  max_seq_length=max_label_length
										  )
			processed_labels.append(label_tokens["token_idxs"])
		
		label_token_idxs = torch.tensor(processed_labels, dtype=torch.long)
		
		tensor_dataset = TensorDataset(label_token_idxs)
		return tensor_dataset
	except Exception as e:
		embed()
		raise e


# def build_index(index_type, label_embeddings):
#
# 	vector_size = label_embeddings.size(1)
# 	index_buffer = params["index_buffer"]
# 	if index_type == "hnsw":
# 		logger.info("Using HNSW index in FAISS")
# 		index = DenseHNSWFlatIndexer(vector_size, index_buffer)
# 	elif:
# 		logger.info("Using Flat index in FAISS")
# 		index = DenseFlatIndexer(vector_size, index_buffer)
#
# 	logger.info("Building index.")
# 	index.index_data(label_embeddings.numpy())
# 	logger.info("Done indexing data.")



def run_inference(dataset, model, k, out_dir):
	
	try:

		assert isinstance(dataset, XMCDataset)
		# Embed all labels
		
		model.eval()
		config = model.config
		tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.lowercase)
		
		tokenized_label_tensor_data = process_label_data(labels=dataset.labels,
														 tokenizer=tokenizer,
														 max_label_length=config.max_label_len)
			
		sampler = SequentialSampler(tokenized_label_tensor_data)
		
		label_dataloader = DataLoader(dataset=tokenized_label_tensor_data,
									  sampler=sampler,
									  batch_size=config.eval_batch_size)
		
		LOGGER.info("Computing label embeddings")
		all_label_embs_list = []
		for label_batch in tqdm(label_dataloader):
			curr_labels = label_batch[0].to(model.device)
			batch_emb = model.encode_label(curr_labels).cpu()
			all_label_embs_list.append(batch_emb)
		
		all_label_embs = torch.cat(all_label_embs_list, dim=0)
		
		# all_label_embs = torch.nn.functional.normalize(all_label_embs)
		
		LOGGER.info("Computed all embeddings")
		# build_index(all_label_embs)
		# index = faiss.IndexFlatL2(d)
		#
		# index.add(all_label_embs)
		
		# LOGGER.info("Embedded all labels")
		# embed()
		
		
		# eval_dataloader = get_dataloader(config=model.config,
		# 								 raw_data=dataset,
		# 								 batch_size=model.config.eval_batch_size)
		
		
		all_label_embs_np = all_label_embs.numpy()
		np.save(arr=all_label_embs_np, file=f"{out_dir}/label_embs")
		
		predictions = []
		gt_labels = []
		# Embed all input data
		
		for datapoint in tqdm(dataset.data): #TODO: Use faiss indexing here for more efficient inference after embedding
			# Predict labels
			data_input = datapoint["input"]
			label_idxs = datapoint["label_idxs"]
			
			input_tokens = tokenize_input(input=data_input,
										  tokenizer=tokenizer,
										  max_seq_length=config.max_input_len)
			
			input_tokens_idxs = torch.tensor(input_tokens["token_idxs"], dtype=torch.long).to(model.device).view(1, -1)
			input_embed = model.encode_input(input_token_idxs=input_tokens_idxs).cpu()
			input_embed = input_embed.squeeze()
			
			# get top-k NN for this input
			
			label_scores = torch.matmul(all_label_embs, input_embed).cpu().detach().numpy()
			
			topk_labels_idxs = np.argpartition(label_scores, -k)[-k:]  # Indices not sorted
			topk_labels_idxs = topk_labels_idxs[np.argsort(label_scores[topk_labels_idxs])][::-1]  # Indices sorted by value from largest to smallest
			
			predictions.append(topk_labels_idxs.tolist())
			gt_labels.append(label_idxs)
			
			# if input("Finished eval") == "1":
			# 	embed()
		
		prec, recall = score_predictions(predictions=predictions, gt_labels=gt_labels, K=k)
		
		LOGGER.info(f'Precision:\n{json.dumps(prec, indent=4)}\n')
		LOGGER.info(f'Recall:\n{json.dumps(recall, indent=4)}\n')
		
		
		return gt_labels, predictions, prec, recall
		
	except Exception as e:
		embed()
		raise e
	

def score_predictions(predictions, gt_labels, K):
	
	prec_at_k = {i: [] for i in range(1, K)}
	recall_at_k = {i: [] for i in range(1, K)}
	
	for curr_gt, curr_pred in zip(gt_labels, predictions):
		for iter_k in range(1, K):
			curr_gt_set = set(curr_gt[:iter_k])
			curr_pred_set = set(curr_pred[:iter_k])
			prec_at_k[iter_k] 	+= [len(curr_pred_set.intersection(curr_gt_set))/iter_k]
			recall_at_k[iter_k] += [len(curr_pred_set.intersection(curr_gt_set))/min(iter_k, len(curr_gt_set))]
			
	
	prec_at_k = {i: float(np.mean(prec_at_k[i])) for i in range(1, K)}
	recall_at_k = {i: float(np.mean(prec_at_k[i])) for i in range(1, K)}
	
	return prec_at_k, recall_at_k
	
	

def main(arg_list):
	parser = argparse.ArgumentParser( description='Eval a pretrained multi-label model')
	
	parser.add_argument("--test_file", type=str, required=True, help="test file")
	parser.add_argument("--lbl_file", type=str, required=True, help="label file")
	parser.add_argument("--model_config_file", type=str, required=True, help="model config file")
	parser.add_argument("--out_dir", type=str, required=True, help="output result dir")
	
	args = parser.parse_args(arg_list)
	
	test_file = args.test_file
	lbl_file = args.lbl_file
	model_config_file = args.model_config_file
	out_dir = args.out_dir
	
	Path(out_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present

	
	
	with open(model_config_file, "r") as f:
		config = json.load(f)
		model = BiEncoderWrapper.load_model(config=config)
	
	
	LOGGER.info("Done loading models")
	
	dataset = read_xmc_data(input_file=test_file, lbl_file=lbl_file, lower_case=model.config.lowercase)
	LOGGER.info("Done loading dataset")
	
	k = 10
	with torch.no_grad():
		gt_labels, predictions, prec, recall = run_inference(dataset=dataset, model=model, k=k, out_dir=out_dir)
		
		test_file_base_name = test_file.split("/")[-1]
		with open(f'{out_dir}/{test_file_base_name}_eval_result.json', "w") as writer:
			result = {"prec_at_k": prec, "recall_at_k": recall}
			json.dump(result, writer)
		
		with open(f'{out_dir}/{test_file_base_name}_predictions.json', "w") as writer:
			for curr_gt, curr_pred in zip(gt_labels, predictions):
				writer.write(json.dumps({"gt_labels": curr_gt, "curr_pred": curr_pred}) + "\n")
		
		# if input("Finished eval") == "1":
		# 	embed()
		#

if __name__ == '__main__':
	main(sys.argv[1:])
