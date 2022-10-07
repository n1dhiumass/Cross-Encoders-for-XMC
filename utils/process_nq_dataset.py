import os
import sys
import json
import logging
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from IPython import embed
from pathlib import Path
from transformers import BertTokenizer

from models.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def add_cls_token(data_split):
	data_dir = "../../data/dpr/downloads/data/retriever_results/nq/single"
	tokenizer_name =  "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
	tknzd_q_file = f"{data_dir}/tokenized_questions_nq_{data_split}.json"
	tknzd_psg_file = f"{data_dir}/tokenized_passage_groups_nq_{data_split}.json"
	# Read Tokenized questions and pad/truncate them to given max_input_len
	
	LOGGER.info("Adding cls token to questions")
	with open(tknzd_q_file, "r") as fin:
		dump_dict_q = json.load(fin)
		all_tknzd_ques = dump_dict_q["data"]
		
		all_tknzd_ques = [[tokenizer.cls_token] + tkn_ids for tkn_ids in all_tknzd_ques]
		

	LOGGER.info("Adding cls token to passages")
	# Tokenize all passages
	with open(tknzd_psg_file, "r") as fin:
		dump_dict_p = json.load(fin)
		all_tknzd_passages = dump_dict_p["data"]
		
		all_tknzd_passages_padded = []
		for curr_tknzd_psg_group in tqdm(all_tknzd_passages):
			
			curr_tknzd_psg_group = [[tokenizer.cls_token] + tkn_ids for tkn_ids in curr_tknzd_psg_group]
			all_tknzd_passages_padded += [curr_tknzd_psg_group]
		
		
	LOGGER.info("Saving questions")
	with open(tknzd_q_file, "w") as fout:
		dump_dict = {
			"misc": dump_dict_q["misc"],
			"data": all_tknzd_ques,
		}
		json.dump(dump_dict, fout, indent=4)
		
	
	LOGGER.info("Saving passages")
	with open(tknzd_psg_file, "w") as fout:
		dump_dict = {
			"misc": dump_dict_p["misc"],
			"data": all_tknzd_passages_padded,
		}
		json.dump(dump_dict, fout, indent=4)


def convert_tokens_to_token_ids(data_split):
	data_dir = "../../data/dpr/downloads/data/retriever_results/nq/single"
	tokenizer_name =  "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
	tknzd_q_file = f"{data_dir}/tokenized_questions_nq_{data_split}.json"
	tknzd_psg_file = f"{data_dir}/tokenized_passage_groups_nq_{data_split}.json"
	# Read Tokenized questions and pad/truncate them to given max_input_len
	
	LOGGER.info(f"Converting token to token_ids for questions for {data_split}")
	with open(tknzd_q_file, "r") as fin:
		dump_dict_q = json.load(fin)
		all_tknzd_ques = dump_dict_q["data"]
		
		all_tknzd_ques = [tokenizer.convert_tokens_to_ids(tkn_ids) for tkn_ids in all_tknzd_ques]
		

	LOGGER.info("Converting token to token_ids for passages")
	# Tokenize all passages
	with open(tknzd_psg_file, "r") as fin:
		dump_dict_p = json.load(fin)
		all_tknzd_passages = dump_dict_p["data"]
		
		all_tknzd_passages_padded = []
		for curr_tknzd_psg_group in tqdm(all_tknzd_passages):
			
			curr_tknzd_psg_group = [tokenizer.convert_tokens_to_ids(tkn_ids) for tkn_ids in curr_tknzd_psg_group]
			all_tknzd_passages_padded += [curr_tknzd_psg_group]
		
		
	LOGGER.info("Saving questions")
	with open(tknzd_q_file, "w") as fout:
		dump_dict = {
			"misc": dump_dict_q["misc"],
			"data": all_tknzd_ques,
		}
		json.dump(dump_dict, fout, indent=4)
		
	
	LOGGER.info("Saving passages")
	with open(tknzd_psg_file, "w") as fout:
		dump_dict = {
			"misc": dump_dict_p["misc"],
			"data": all_tknzd_passages_padded,
		}
		json.dump(dump_dict, fout, indent=4)
		

def main(data_split):
	
	data_dir = "../../data/dpr/downloads/data/retriever_results/nq/single"
	filename = f"{data_dir}/{data_split}.json"
	tokenizer_name =  "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
	
	LOGGER.info(f"Reading data for split = {data_split} from file {filename}")
	with open(filename, "r") as fin:
		raw_data = json.load(fin)
		
	
	all_questions = []
	all_passage_groups = []
	
	# "question": "when is the next deadpool movie being released",
    # "answers": [
    #         "May 18 , 2018"
    #     ],
	# "ctxs":[
	# {
    #             "id": "18960839",
    #             "title": "Deadpool 2",
    #             "text": "is dedicated to her memory. The film's score is the first to receive a parental advisory warning for explicit content, and the soundtrack also includes the original song \"Ashes\" by C\u00e9line Dion. \"Deadpool 2\" was released in the United States on May 18, 2018. It has grossed over $738 million worldwide, becoming the seventh highest-grossing film of 2018, as well as the third highest-grossing R-rated film and the third highest-grossing \"X-Men\" film. It received positive reviews from critics, who praised its humor, acting (particularly Reynolds, Brolin, and Beetz's performances), story, and action sequences, with some calling it better than the",
    #             "score": "83.40279",
    #             "has_answer": true
    # }],
	
	for q_w_passages in raw_data:
		question = q_w_passages["question"]
		passages = q_w_passages["ctxs"]
		
		all_questions += [question]
		all_passage_groups += [passages]
	
	
	# Tokenize questions
	LOGGER.info(f"Tokenizing {len(all_questions)} questions")
	all_tknzd_ques = [[tokenizer.cls_token] + tokenizer.tokenize(question) for question in tqdm(all_questions)]
	all_tknzd_ques = [tokenizer.convert_tokens_to_ids(q_tkns) for q_tkns in tqdm(all_tknzd_ques)]
	
	
	# Tokenize passages
	all_tknzd_psg_groups = []
	LOGGER.info(f"Tokenizing {len(all_questions)} passages")
	for curr_passage_group in tqdm(all_passage_groups):
		
		curr_tknzd_psg_group = []
		for passage in curr_passage_group:
			title = passage["title"]
			text = passage["text"]
			tknzd_psg = [tokenizer.cls_token] + tokenizer.tokenize(title) + [ENT_TITLE_TAG] + tokenizer.tokenize(text)
			curr_tknzd_psg_group += [tokenizer.convert_tokens_to_ids(tknzd_psg)]
		
		all_tknzd_psg_groups += [curr_tknzd_psg_group]
	
	
	misc_info = {
		"source_file": filename,
		"tokenizer": tokenizer_name,
		"notes": f"Used {ENT_TITLE_TAG} for concatenating passage title and text, no truncation of tokens"
	}
	
	with open(f"{data_dir}/tokenized_questions_nq_{data_split}.json", "w") as fout:
		dump_dict = {
			"misc": misc_info,
			"data": all_tknzd_ques,
		}
		json.dump(dump_dict, fout, indent=4)
		
	
	with open(f"{data_dir}/tokenized_passage_groups_nq_{data_split}.json", "w") as fout:
		dump_dict = {
			"misc": misc_info,
			"data": all_tknzd_psg_groups,
		}
		json.dump(dump_dict, fout, indent=4)
		
	pass





if __name__ == "__main__":
	main(data_split="dev")
	# convert_tokens_to_token_ids(data_split="test")
	# convert_tokens_to_token_ids(data_split="dev")
	# convert_tokens_to_token_ids(data_split="train")
	

