import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from collections import defaultdict
from pytorch_transformers.tokenization_bert import BertTokenizer
from eval.run_gradient_based_search_w_cross_enc import get_token_pairs_by_position, discretize_soft_sequence, discretize_soft_sequence_wo_crf

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)








def test_cases():
	
	seq_len, vocab_len = 4, 7
	all_vals = list(range(vocab_len))
	
	# # Uniform emission and transition probabilities
	# allowed_prev_vals = {
	# 	0: {val:all_vals for val in all_vals},
	# 	1: {val:all_vals for val in all_vals},
	# 	2: {val:all_vals for val in all_vals},
	# 	3: {val:all_vals for val in all_vals},
	# }
	# emission_probs = {
	# 	0: {val:ctr for ctr, val in enumerate(all_vals)},
	# 	1: {val:ctr for ctr, val in enumerate(all_vals)},
	# 	2: {val:ctr for ctr, val in enumerate(all_vals)},
	# 	3: {val:ctr for ctr, val in enumerate(all_vals)},
	# }
	
	# Only allow transition bw/ (v,v+1) and (0,0)
	allowed_prev_vals = {
		i: {val:[_v for _v in all_vals if _v < val or _v == 0] for val in all_vals} for i in range(1, seq_len)
	}
	# Emission prob is proportional to token idx
	emission_probs = {
		i: {val:val for ctr, val in enumerate(all_vals)} for i in range(seq_len)
	}
	# Best sequences should be increasing eg [3,4,5,6] for seq_len, vocab_len = 4, 7 with score = 3+4+5+6=18
	
	
	# # Only allow transition bw/ (v+1,v) and (,0)
	# allowed_prev_vals = {
	# 	i: {val:[_v for _v in all_vals if _v > val or _v == seq_len-1] for val in all_vals} for i in range(1, seq_len)
	# }
	# # Emission prob is proportional to -ve token idx
	# emission_probs = {
	# 	i: {val:-val for ctr, val in enumerate(all_vals)} for i in range(seq_len)
	# }
	# # Best sequences should be increasing eg [3,2,1,0] for seq_len, vocab_len = 4, 7 with score = -3-2-1-0=-6
	
	discretize_soft_sequence(
		allowed_prev_vals=allowed_prev_vals,
		emission_probs=emission_probs,
		seq_len=seq_len,
		val_vocab=all_vals
	)


def get_token_probs(token_embeds, tokens, curr_embed):
	scores = curr_embed*token_embeds
	
	res = {token_id:score for token_id, score in zip(tokens, scores)}
	
	return res
	

def main():
	try:
		data_dir = "../../data/zeshel"
		domain = "yugioh"
		entity_tokens_file = f"{data_dir}/tokenized_entities/{domain}_128_bert_base_uncased.npy"
		
		# init tokenizer
		tokenizer = BertTokenizer.from_pretrained(
			"bert-base-uncased", do_lower_case=True
		)
		
		entity_tokens = np.load(entity_tokens_file)
		n_ents, max_seq_len = entity_tokens.shape
		val_vocab = tokenizer.convert_tokens_to_ids(list(tokenizer.vocab.keys()))
		
		
		
		allowed_prev_vals = get_token_pairs_by_position(entity_tokens=entity_tokens)
		
	
		emission_probs = {i:{v:1 for v in val_vocab} for i in range(max_seq_len)}
		
		
		# This should decode entity_0
		emission_probs = {i:{entity_tokens[0][i]:1} for i in range(max_seq_len)}
		
		
		emission_probs = {
			i:{entity_tokens[0][i]:1} for i in range(max_seq_len)
		}
		
		
		LOGGER.info(f"Entity zero = {entity_tokens[0]}")
		
		
		discretize_soft_sequence(
			allowed_prev_vals=allowed_prev_vals,
			emission_probs=emission_probs,
			seq_len=max_seq_len,
			val_vocab=val_vocab
		)
	
	
		
	except Exception as e:
		embed()
		raise e

	
	
if __name__ == "__main__":
	main()
	# test_cases()
