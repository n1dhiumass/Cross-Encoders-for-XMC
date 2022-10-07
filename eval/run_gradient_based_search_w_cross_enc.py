import os
import sys
import json
import wandb
import argparse

from tqdm import tqdm
import logging
import torch
import itertools
import numpy as np
from IPython import embed
from pathlib import Path
from os.path import dirname
from collections import defaultdict

from pytorch_transformers.modeling_bert import BertModel, BertEmbeddings
from pytorch_transformers.tokenization_bert import BertTokenizer

from eval.eval_utils import compute_label_embeddings
from models.crossencoder import CrossEncoderWrapper
from models.biencoder import BiEncoderWrapper
from utils.data_process import load_entities, load_mentions, get_context_representation
from utils.optimizer import get_bert_optimizer
from models.nearest_nbr import build_flat_or_ivff_index
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_PAIR_LENGTH, MAX_MENT_LENGTH, MAX_ENT_LENGTH
from utils.config import GradientBasedInfConfig



logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


class CustomLabelEmbeddings(BertEmbeddings):
	"""
	Base Class that supports use of various ways of computing and parameterizing label token embeddings
	"""
	def __init__(self, config, label_start, label_len, embed_dim):
		super(CustomLabelEmbeddings, self).__init__(config=config)
		
		# TODO: Make sure that token_type_ids are 0000...000111...111 format and there are no trainling zeros
		# TODO: When we are optimizing, we should allow attention mask and token_type_ids to be flexible as well so that we can have flexibility in terms of entities that we are predicting
		self.label_start = label_start
		self.label_len = label_len
		self.embed_dim = embed_dim
		
		
	
	@property
	def device(self):
		return self.word_embeddings.weight.device
	
	def get_label_token_embed(self):
		raise NotImplementedError
	
	def forward(self, input_ids, token_type_ids=None, position_ids=None):
		"""
		Forward function of BertEmbeddings class modified to allow for custom label token embeddings to support
		gradient-based inference
		:param input_ids:
		:param token_type_ids:
		:param position_ids:
		:return:
		"""
		
		seq_length = input_ids.size(1)
		if position_ids is None:
			position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)


		words_embeddings = self.word_embeddings(input_ids)
		
		################################################################################################################
		# Instead of indexing in an embedding array for label tokens embeddings, just use custom label token embeddings
		custom_label_embed = self.get_label_token_embed()
		words_embeddings[:, self.label_start: ] = custom_label_embed
		################################################################################################################
		
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		assert words_embeddings.shape == position_embeddings.shape
		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		
		return embeddings


class EmbeddingWFreeParams(CustomLabelEmbeddings):
	"""
	Class that supports use of "trainable" embeddings for the label in a cross-encoder model.
	This takes in the input pair and return the embedding for the input token sequence, with embedding of the label
	tokens coming from separate trainable embeddings (i.e. free parameters).
	"""
	def __init__(self, config, label_start, label_len, embed_dim, state_dict):
		super(EmbeddingWFreeParams, self).__init__(
			config=config,
			label_start=label_start,
			label_len=label_len,
			embed_dim=embed_dim
		)
		# Free-parameters for label tokens
		self.label_token_embeds = torch.nn.Embedding(num_embeddings=label_len, embedding_dim=embed_dim)
		
		# Load other parameters from given state_dict
		missing_keys, unexp_keys = self.load_state_dict(state_dict=state_dict, strict=False)
		assert len(unexp_keys) == 0, f"Following keys are unexpected in state_dict {unexp_keys}"
		assert len(missing_keys) == 1, f"Exactly one key is missing in state_dict = {missing_keys}"
		assert missing_keys[0] == "label_token_embeds.weight", f"Expected missing key = label_token_embeds.weight but found missing key = {missing_keys}"
		
	
	def init_label_embeds_from_word_embeds(self, label_tokens):
		"""
		Use word_embeddings for given tokens to initialize label_token_embeds parameters
		:param label_tokens:
		:return:
		"""
		#TODO: Make sure that label embeds and word embeds DO NOT share memory
		assert len(label_tokens) == self.label_token_embeds.weight.shape[0], f"Label Tokens shape = {len(label_tokens)} != Label embeds shape {self.label_token_embeds.weight.shape[0]}"
		label_tokens = label_tokens if torch.is_tensor(label_tokens) else torch.LongTensor(label_tokens)
		label_embs = self.word_embeddings(label_tokens.to(self.device))
		self.label_token_embeds.weight.data = label_embs
	
	def init_label_embeds_from_given_embeds(self, label_token_embeds):
		"""
		Directly use given label_token_embeds to initialize label_token_embeds parameters
		:param label_token_embeds:
		:return:
		"""
		assert label_token_embeds.shape == self.label_token_embeds.weight.shape, f"Given Label Embeds shape = {label_token_embeds.shape} != Label embeds shape {self.label_token_embeds.weight.shape}"
		assert label_token_embeds.shape == (self.label_len, self.embed_dim)
		self.label_token_embeds.weight.data = label_token_embeds.to(self.device)
	
	def get_label_token_embed(self):
		return self.label_token_embeds.weight


class SoftConvexTokenEmbedding(CustomLabelEmbeddings):
	"""
	Class that supports use of weights to combine fixed token embeddings to generate a "token" embedding for
	each token position for the label in a cross-encoder model.
	This takes in the input pair and return the embedding for the input token sequence, with embedding of the label
	tokens coming from a learned weighted combination of word embeddings.
	"""
	def __init__(self, config, label_start, label_len, embed_dim, state_dict, allowed_tokens_per_pos):
		super(SoftConvexTokenEmbedding, self).__init__(
			config=config,
			label_start=label_start,
			label_len=label_len,
			embed_dim=embed_dim
		)
		
		# self.soft_word_embeddings_bag = torch.nn.EmbeddingBag.from_pretrained(embeddings=self.word_embeddings.weight, freeze=True, mode="sum")
		self.soft_word_embeddings = torch.nn.Embedding.from_pretrained(embeddings=self.word_embeddings.weight, freeze=True)
		
		"""
		Variables for storing weights for tokens allowed at each position.
		soft_label_token_idxs_per_pos : Stores list of allowed tokens for all positions
		soft_label_token_weights_per_pos : Stores list of tokens weights for all positions
		soft_label_token_offsets : Stores index pointer for each position in label. Shape = (label_len,)
		
		For position = 0 to label_len - 2
			list of allowed tokens => soft_label_token_idxs_per_pos[ soft_label_token_offsets[i] : soft_label_token_offsets[i+1] ]
			weights of allowed tokens => soft_label_token_weights_per_pos[ soft_label_token_offsets[i] : soft_label_token_offsets[i+1] ]
		
		For last position = label_len - 1,
			list of allowed tokens => soft_label_token_idxs_per_pos[ soft_label_token_offsets[i] : ]
			weights of allowed tokens => soft_label_token_weights_per_pos[ soft_label_token_offsets[i] : ]
			
		Defining soft_label_token_idxs_per_pos and soft_label_token_offsets using nn.Parameters so that they are moved
		to the right device with the class
		"""
		self.soft_label_token_idxs_per_pos = torch.nn.Parameter(torch.LongTensor([_token for i in range(label_len) for _token in allowed_tokens_per_pos[i]], device=self.device), requires_grad=False)
		self.soft_label_token_offsets = np.cumsum([len(allowed_tokens_per_pos[i]) for i in range(label_len)]).tolist()
		self.soft_label_token_offsets = torch.nn.Parameter(torch.LongTensor([0] + self.soft_label_token_offsets[:-1], device=self.device), requires_grad=False)
		
		self.soft_label_token_weights_per_pos = torch.nn.Parameter(torch.empty(self.soft_label_token_idxs_per_pos.shape[0], device=self.device), requires_grad=True)
		torch.nn.init.uniform_(self.soft_label_token_weights_per_pos) # TODO: Add support for other initializations
		
		
		# Load other parameters from given state_dict
		missing_keys, unexp_keys = self.load_state_dict(state_dict=state_dict, strict=False)
		assert len(unexp_keys) == 0, f"Following keys are unexpected in state_dict {unexp_keys}"
		assert len(missing_keys) == 4, f"Exactly four keys should missing in state_dict = {missing_keys}"
		assert set(missing_keys).__eq__({'soft_label_token_idxs_per_pos', 'soft_label_token_offsets', 'soft_label_token_weights_per_pos', 'soft_word_embeddings.weight'}), f"Expected missing key = 'soft_label_token_idxs_per_pos', 'soft_label_token_offsets', 'soft_label_token_weights_per_pos', 'soft_word_embeddings.weight' but found missing key = {missing_keys}"
		
		assert len(self.soft_label_token_offsets) == label_len, f"len(self.soft_label_token_offsets) = {len(self.soft_label_token_offsets)} != label_len = {label_len}"
		assert self.soft_label_token_idxs_per_pos.shape == self.soft_label_token_weights_per_pos.shape, f"self.soft_label_token_idxs_per_pos.shape = {self.soft_label_token_idxs_per_pos.shape} != self.soft_label_token_weights_per_pos.shape = {self.soft_label_token_weights_per_pos.shape}"
	
	
	def init_soft_tokens_w_given_label(self, label_tokens, smooth_alpha):
	
		try:
			assert smooth_alpha != 0.0, "It is not possible to assign exactly zero probability to a valid token but we can assign extremely small negligible probablity mass. Consider using small values of smooth_alpha such as 1e-10"
			assert 0 < smooth_alpha < 1, "This parameter should be b/w 0 and 1 (end points excluded)"
			
			with torch.no_grad(): # no_grad required as we are using in-place operation on parameters that require grad
				
				label_tokens = label_tokens.cpu().numpy().tolist() if torch.is_tensor(label_tokens) else label_tokens
				label_tokens = label_tokens.tolist() if isinstance(label_tokens, np.ndarray) else label_tokens
				assert isinstance(label_tokens, list)
				for i in range(self.label_len):
					start_idx = self.soft_label_token_offsets[i]
					end_idx = self.soft_label_token_offsets[i+1] if i < self.label_len - 1 else None
					
					# List of allowed tokens at position i
					curr_allowed_tokens = self.soft_label_token_idxs_per_pos[start_idx:end_idx]
					
					# Token which gets most probablity mass at position i (modulo smoothening parameter)
					curr_token = label_tokens[i]
					assert isinstance(curr_token, int)
					
					curr_weights = (1 - smooth_alpha)*(curr_allowed_tokens == curr_token) # Give 1-smooth_alpha probability to curr_token
					if len(curr_allowed_tokens) >= 1: # Distributed smooth_alpha probability mass over rest of the tokens
						curr_weights += (smooth_alpha/len(curr_allowed_tokens))
					
					assert torch.sum((curr_allowed_tokens == curr_token)) == 1, f"curr_token = {curr_token} should occur exactly once in curr_allowed_tokens = {curr_allowed_tokens}"
					assert np.round(torch.sum(curr_weights).cpu().numpy(), decimals=2) == 1., f"Probability mass = {torch.sum(curr_weights)} but it should sum up to 1"
					
					curr_weights_log = torch.log(curr_weights)
					
					self.soft_label_token_weights_per_pos[start_idx:end_idx] = curr_weights_log
			
		except Exception as e:
			embed()
			raise e
	
			
	def get_label_token_embed(self):
		
		try:
			softmax = torch.nn.Softmax(dim=0)
			per_sample_weights = []
			curr_label_embed = []
			for i in range(self.label_len):
				start_idx = self.soft_label_token_offsets[i]
				end_idx = self.soft_label_token_offsets[i+1] if i < self.label_len - 1 else None
				
				weights = softmax(self.soft_label_token_weights_per_pos[start_idx:end_idx])
				embeds = self.soft_word_embeddings(self.soft_label_token_idxs_per_pos[start_idx:end_idx])
				wgtd_embed = torch.matmul(weights,embeds)
				
				per_sample_weights += [weights]
				curr_label_embed += [wgtd_embed]
			
			curr_label_embed = torch.stack(curr_label_embed)
			assert curr_label_embed.shape == (self.label_len, self.embed_dim), f"curr_label_embed.shape = {curr_label_embed.shape} != (self.label_len, self.embed_dim) = {(self.label_len, self.embed_dim)}"
			# per_sample_weights = torch.concat(per_sample_weights)
			# assert per_sample_weights.shape == self.soft_label_token_weights_per_pos.shape, f" Normalized weight shape = {per_sample_weights.shape} != unnormalized weight shape = {self.soft_label_token_weights_per_pos.shape}"
			#
			#
			# # Compute label embeddings using learned weighted average of word embeddings
			# curr_label_embed2 = self.soft_word_embeddings_bag(
			# 	input=self.soft_label_token_idxs_per_pos.detach(),
			# 	offsets=self.soft_label_token_offsets.detach(),
			# 	per_sample_weights=per_sample_weights.detach()
			# )
	
			return curr_label_embed
		except Exception as e:
				embed()
				raise e
	

	def get_per_pos_token_probs(self):
		"""
		
		:return: Dictionary mapping position index to dict containing token index and corresponding probability
		"""
		softmax = torch.nn.Softmax(dim=0)
		per_pos_weights = {}
		for i in range(self.label_len):
			start_idx = self.soft_label_token_offsets[i]
			end_idx = self.soft_label_token_offsets[i+1] if i < self.label_len - 1 else None
			
			indices = self.soft_label_token_idxs_per_pos[start_idx:end_idx].cpu().numpy().tolist()
			weights = softmax(self.soft_label_token_weights_per_pos[start_idx:end_idx]).detach().cpu().numpy().tolist()
	
			per_pos_weights[i] = {
				idx:w for idx,w in zip(indices, weights)
			}
			
		return per_pos_weights
	
		
class SoftConvexLabelEmbedding(CustomLabelEmbeddings):
	"""
	Class that supports use of single set of weights to combine token embeddings of labels instead of using separate
	per-position weights as done in SoftConvexTokenEmbedding
	This takes in the input pair and return the embedding for the input token sequence, with embedding of the label
	tokens coming from a learned weighted combination of word embeddings.
	"""
	def __init__(self, config, label_start, label_len, embed_dim, state_dict, allowed_tknzd_labels):
		super(SoftConvexLabelEmbedding, self).__init__(
			config=config,
			label_start=label_start,
			label_len=label_len,
			embed_dim=embed_dim
		)
		
		self.num_labels = allowed_tknzd_labels.shape[0]
		
		self.allowed_tknzd_labels = torch.nn.Parameter(torch.LongTensor(allowed_tknzd_labels), requires_grad=False)
		
		# Define weight for n_labels
		self.soft_label_token_weights = torch.nn.Parameter(torch.empty(self.num_labels, device=self.device), requires_grad=True)
		torch.nn.init.uniform_(self.soft_label_token_weights) # TODO: Add support for other initializations
		
		
		# Load other parameters from given state_dict
		missing_keys, unexp_keys = self.load_state_dict(state_dict=state_dict, strict=False)
		assert len(unexp_keys) == 0, f"Following keys are unexpected in state_dict {unexp_keys}"
		assert len(missing_keys) == 2, f"Exactly two keys should missing in state_dict = {missing_keys}"
		assert set(missing_keys).__eq__({'allowed_tknzd_labels', 'soft_label_token_weights'}), f"Expected missing key = 'allowed_tknzd_labels', 'soft_label_token_weights' but found missing key = {missing_keys}"
		assert self.allowed_tknzd_labels.shape == (self.num_labels, self.label_len), f"shape mismatch:: self.allowed_tknzd_labels.shape = {self.allowed_tknzd_labels.shape} != (self.num_labels, self.label_len) = {(self.num_labels, self.label_len)}"
		
		assert self.soft_label_token_weights.shape == (self.num_labels,), f"self.soft_label_token_weights.shape = {self.soft_label_token_weights.shape} != self.num_labels = {self.num_labels} "
	
		
	def init_soft_tokens_w_given_label(self, label_idx, smooth_alpha):
	
		
		assert smooth_alpha != 0.0, "It is not possible to assign exactly zero probability to a valid token but we can assign extremely small negligible probablity mass. Consider using small values of smooth_alpha such as 1e-10"
		assert 0 < smooth_alpha < 1, "This parameter should be b/w 0 and 1 (end points excluded)"
		
		curr_weights  = torch.zeros(self.num_labels)
		if self.num_labels >= 1: # Distributed smooth_alpha probability mass over all labels
			curr_weights += smooth_alpha/self.num_labels
		curr_weights[label_idx] += (1 - smooth_alpha) # Give additional 1-smooth_alpha probability to given label_idx
		
		assert np.round(torch.sum(curr_weights).cpu().numpy(), decimals=2) == 1., f"Probability mass = {torch.sum(curr_weights)} but it should sum up to 1"
		
		curr_weights_log = torch.log(curr_weights).to(self.soft_label_token_weights.device)
		self.soft_label_token_weights.data = curr_weights_log
		

	def get_label_weights(self):
		"""
		:return: (self.num_labels,) shape tensor containing weight for each label
		"""
		softmax = torch.nn.Softmax(dim=0)
		weights = softmax(self.soft_label_token_weights)
		return weights

		
	def get_label_token_embed(self):
		
		weights = self.get_label_weights()
		
		curr_label_embed = []
		for i in range(self.label_len):
			torch.cuda.empty_cache()
			
			curr_pos_all_token_embeds = self.word_embeddings(self.allowed_tknzd_labels[:, i])
			num_labels, embed_dim = curr_pos_all_token_embeds.shape
			
			curr_pos_token_embed = torch.matmul(weights, curr_pos_all_token_embeds)
			curr_label_embed += [curr_pos_token_embed]
			
			assert num_labels == self.num_labels
		
		curr_label_embed = torch.stack(curr_label_embed)
		assert curr_label_embed.shape == (self.label_len, self.embed_dim), f"curr_label_embed.shape = {curr_label_embed.shape} != (self.label_len, self.embed_dim) = {(self.label_len, self.embed_dim)}"
		
		return curr_label_embed


class InputLabelPairBert(BertModel):
	"""
	BertModel with custom embedding object. This allows us fine grained control on types of embeddings fed into BERT
	and also gives us the flexibility to use separate learned parameters as part of input to BERT
	"""
	def __init__(self, config, state_dict, param_type, label_start, label_len, embed_dim, allowed_tokens_per_pos, allowed_tknzd_labels):
		"""
		
		:param config: Config for BertModel
		:param state_dict: BertModel state dict
		:param param_type: Method to parameterize the model for gradient-based inference
		:param label_start: Start_idx for label tokens in the instance-label pair given ot the model
		:param label_len: number of label tokens
		:param embed_dim:
		:param allowed_tokens_per_pos:
		:param allowed_tknzd_labels:
		"""
		super(InputLabelPairBert, self).__init__(config=config)
		
		# Load parameters for parent BERT model
		self.load_state_dict(state_dict=state_dict)
		
		self.default_embeddings = self.embeddings # Save pointer to original default embeddings
		default_emb_state_dict = self.embeddings.state_dict()
		
		# Initialize custom embedding parameters
		if param_type == "free_embeds":
			LOGGER.info("Using free parameters for each label token instead of using token embeddings from BertEmbeddings")
			self.custom_embeddings = EmbeddingWFreeParams(
				config=config,
				label_len=label_len,
				label_start=label_start,
				embed_dim=embed_dim,
				state_dict=default_emb_state_dict
			)
		
		elif param_type == "per_pos_weights":
			assert allowed_tokens_per_pos is not None
			LOGGER.info("Using weighted combination of token embeddings for each position")
			self.custom_embeddings = SoftConvexTokenEmbedding(
				config=config,
				label_len=label_len,
				label_start=label_start,
				embed_dim=embed_dim,
				state_dict=default_emb_state_dict,
				allowed_tokens_per_pos=allowed_tokens_per_pos
			)
		
		elif param_type == "per_label_weight":
			assert allowed_tknzd_labels is not None
			LOGGER.info("Using weighted combination of labels to get label token embeddings for each position")
			self.custom_embeddings = SoftConvexLabelEmbedding(
				config=config,
				label_len=label_len,
				label_start=label_start,
				embed_dim=embed_dim,
				state_dict=default_emb_state_dict,
				allowed_tknzd_labels=allowed_tknzd_labels
			)
			
		else:
			raise NotImplementedError(f"param_type = {param_type} not implemented")
		
		self.embeddings = self.custom_embeddings # Point embeddings variable to custom embeddings


class CustomLabelBert(BertModel):
	"""
	BertModel with custom embedding object. This allows us fine grained control on types of embeddings fed into BERT
	and also gives us the flexibility to use separate learned parameters as part of input to BERT
	"""
	def __init__(self, config, state_dict, label_len, embed_dim):
		super(CustomLabelBert, self).__init__(config=config)
		
		# Load parameters for parent BERT model
		self.load_state_dict(state_dict=state_dict)
		
		default_emb_state_dict = self.embeddings.state_dict()
		# Initialize embedding parameters. label_start = 0 because entire input is a label
		self.custom_embeddings = EmbeddingWFreeParams(
			config=config,
			label_len=label_len,
			label_start=0,
			embed_dim=embed_dim,
			state_dict=default_emb_state_dict
		)
		
		# Additional pointer to new embeddings parameters to facilitate switching b/w custom and default embeddings
		self.default_embeddings = self.embeddings
		
		


class GradientBasedInference(object):
	
	def __init__(self, config):
		
		try:
			assert isinstance(config, GradientBasedInfConfig)
			self.config = config
			assert isinstance(self.config, GradientBasedInfConfig)
			
			self.lr = self.config.lr
			self.quant_method = self.config.quant_method
			self.dataset_name = self.config.data_name
			
			self.max_ent_length = MAX_ENT_LENGTH
			self.max_ment_length = MAX_MENT_LENGTH
			self.max_pair_length = MAX_PAIR_LENGTH
			
			
			DATASETS = get_dataset_info(data_dir=self.config.data_dir, worlds=get_zeshel_world_info(), res_dir=None)
			self.entity_file = DATASETS[self.config.data_name]["ent_file"]
			self.mention_file = DATASETS[self.config.data_name]["ment_file"]
			self.ent_tokens_file = DATASETS[self.config.data_name]["ent_tokens_file"]
			
			# Variable update in init_data
			self.mention_tokens_list = []
			self.complete_entity_tokens_list = []
			self.gt_labels = []
			self.id2title = {}
			self.label_search_index = None
			
			# Required for viterbi-based discretization of soft sequence
			self.all_token_ids = [] # List of all token ids in vocabulary
			self.allowed_prev_vals = {} # Dict storing info about pairwise occurrence of token ids
			self.token_seq_to_label_idx = {} # Dict mapping label token seq to label index
			self.all_token_embs  = [] # Tensor storing embeddings for all tokens
			self.allowed_tokens_per_pos = {}
			
			
			# Load models
			self.crossencoder, self.biencoder = self.load_models(
				cross_model_file=self.config.cross_model_file,
				bi_model_file=self.config.bi_model_file
			)
			
			# Load data	and process it as required
			self.init_data(
				mention_file=self.mention_file,
				entity_file=self.entity_file,
				ent_tokens_file=self.ent_tokens_file
			)
			
			# Prepare models for gradient-based inference
			self.prepare_model_for_grad_inf(
				crossencoder=self.crossencoder,
				biencoder=self.biencoder,
				param_type=self.config.param_type,
				label_start=self.max_ment_length,
				label_len=self.max_ent_length-1,
				embed_dim=self.config.embed_dim,
				allowed_tokens_per_pos=self.allowed_tokens_per_pos,
				allowed_tknzd_labels=self.complete_entity_tokens_list[:, 1:] # Remove cls tokens
			)
			
			# Build label index
			self.label_search_index = self.build_label_search_index(
				quant_method=self.config.quant_method,
				all_label_tokens=self.complete_entity_tokens_list,
				crossencoder=self.crossencoder,
				biencoder=self.biencoder
			)
	
			
		except Exception as e:
			LOGGER.info(f"Error raised {str(e)}")
			embed()
			raise e
	
	
	@staticmethod
	def load_models(cross_model_file, bi_model_file):
		
		if cross_model_file.endswith(".json"):
			with open(cross_model_file, "r") as fin:
				config = json.load(fin)
				crossencoder = CrossEncoderWrapper.load_model(config=config)
		else:
			crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
		
		crossencoder.eval()
		
		
		if os.path.isfile(bi_model_file):
			if bi_model_file.endswith(".json"):
				with open(bi_model_file, "r") as fin:
					config = json.load(fin)
					biencoder = BiEncoderWrapper.load_model(config=config)
			else:
				biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
			biencoder.eval()
		else:
			biencoder = None
		
		return crossencoder, biencoder
	
	@property
	def num_labels(self):
		return len(self.complete_entity_tokens_list)
	
	def init_data(self, mention_file, entity_file, ent_tokens_file):
		"""
		Read data from files and also compute some additional info required during inference based on data
		:param mention_file: File containing mention data
		:param entity_file: File containing entity data
		:param ent_tokens_file:  File containing tokenized entities
		:return:
		"""
		
		#################################### LOAD ENTITY AND MENTION DATA ##############################################
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=entity_file)
		
		tokenizer = self.crossencoder.tokenizer
		
		self.id2title = id2title
		
		LOGGER.info("Loading test samples")
		test_data = load_mentions(
			mention_file=mention_file,
			kb_id2local_id=kb_id2local_id
		)
		test_data = test_data[:self.config.n_ment]
		
		# test_data = test_data[:n_ment] if n_ment > 0 else test_data
		self.gt_labels = np.array([x["label_id"] for x in test_data])
		
		LOGGER.info(f"Tokenize {len(test_data)} test samples")
		# First extract all mentions and tokenize them
		self.mention_tokens_list = [get_context_representation(sample=mention,
															   tokenizer=tokenizer,
															   max_seq_length=self.max_ment_length)["ids"]
									for mention in tqdm(test_data)]
		
		self.complete_entity_tokens_list = np.load(ent_tokens_file)
		
		
		# Required for viterbi based discretization of soft label token sequence
		# self.complete_entity_tokens_list[:, 1:] --> This is to remove CLS token before computing this dictionary
		entity_tokens_wo_cls = self.complete_entity_tokens_list[:, 1:] # Remove cls token
		LOGGER.info("Converting tokens to ids")
		self.all_token_ids = tokenizer.convert_tokens_to_ids(list(tokenizer.vocab.keys()))
		LOGGER.info("Finished converting tokens to ids")
		self.allowed_prev_vals = get_token_pairs_by_position(entity_tokens=entity_tokens_wo_cls)
		self.token_seq_to_label_idx = {tuple(curr_ent_tokens.tolist()):idx for idx, curr_ent_tokens in enumerate(entity_tokens_wo_cls)}
		
		self.allowed_tokens_per_pos = get_allowed_tokens_by_position(entity_tokens=entity_tokens_wo_cls)
		self.all_token_embs = self.crossencoder.model.encoder.bert_model.embeddings.word_embeddings.weight
		
	@staticmethod
	def prepare_model_for_grad_inf(crossencoder, biencoder, param_type, label_start, label_len, embed_dim, allowed_tokens_per_pos, allowed_tknzd_labels):
		"""
		Prepare model parameters for enabling gradient-based inference by changing embedding layer
		:param crossencoder:
		:param biencoder:
		:param param_type
		:param label_start:
		:param label_len:
		:param embed_dim:
		:param allowed_tokens_per_pos:
		:param allowed_tknzd_labels:
		:return:
		"""
		
		config = crossencoder.model.encoder.bert_model.config
		state_dict = crossencoder.model.encoder.bert_model.state_dict()
		new_model = InputLabelPairBert(
			config=config,
			state_dict=state_dict,
			label_start=label_start,
			label_len=label_len,
			embed_dim=embed_dim,
			param_type=param_type,
			allowed_tknzd_labels=allowed_tknzd_labels,
			allowed_tokens_per_pos=allowed_tokens_per_pos
		)
		new_model = new_model.to(crossencoder.device)
		# Replace bert model inside of crossencoder wrapper with this new custom bert model
		crossencoder.model.encoder.bert_model = new_model
		crossencoder.eval()
		
		################ Create custom Biencoder model that supports embedding labels as required ######################
		if biencoder is not None:
			config = biencoder.model.label_encoder.bert_model.config
			state_dict = biencoder.model.label_encoder.bert_model.state_dict()

			new_model = CustomLabelBert(config=config, state_dict=state_dict, label_len=label_len, embed_dim=embed_dim)
			new_model = new_model.to(biencoder.device)

			# Replace label encoder with this new custom encoder model
			biencoder.model.label_encoder.bert_model = new_model
			biencoder.eval()
		
	
	@staticmethod
	def build_label_search_index(quant_method, crossencoder, biencoder, all_label_tokens):
		"""
		Build an nearest nbr search index if required by the given quantization method
		:param quant_method:
		:param crossencoder:
		:param biencoder:
		:param all_label_tokens:
		:return:
		"""
		
		if quant_method in ["viterbi", "unigram_greedy", "label_greedy"]:
			return None
		elif quant_method in ["concat", "bienc"]:
			if not torch.is_tensor(all_label_tokens):
				all_label_tokens = torch.tensor(all_label_tokens)
			
			if quant_method == "concat":
				token_embs = crossencoder.model.encoder.bert_model.embeddings.word_embeddings.weight
				token_embs = token_embs.cpu().detach().numpy()
				
				# Removing first cls token
				all_label_tokens = all_label_tokens[:, 1:]
				label_embs = np.concatenate([token_embs[entity_tokens].reshape(1, -1) for entity_tokens in all_label_tokens])
			elif quant_method == "bienc":
				label_embs = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=all_label_tokens, batch_size=100)
			else:
				raise NotImplementedError(f"Quantization method = {quant_method} not implemented")
			
			LOGGER.info(f"Building index over label embeddings = {label_embs.shape}")
			index = build_flat_or_ivff_index(embeds=label_embs, force_exact_search=False)
			LOGGER.info(f"Finished building index over label embeddings = {label_embs.shape}")
			
			return index
		else:
			raise NotImplementedError(f"Quantization method = {quant_method} not implemented")
		
	
	def init_w_given_label(self, crossencoder, label_idx, label_tokens, smooth_alpha):
		
		assert isinstance(crossencoder.model.encoder.bert_model, InputLabelPairBert)
		if self.config.param_type == "per_label_weight":
			smooth_alpha = smooth_alpha if smooth_alpha is not None else self.config.smooth_alpha

			crossencoder.model.encoder.bert_model.embeddings.init_soft_tokens_w_given_label(
				label_idx=label_idx,
				smooth_alpha=smooth_alpha
			)
		elif self.config.param_type == "per_pos_weights":
			smooth_alpha = smooth_alpha if smooth_alpha is not None else self.config.smooth_alpha

			crossencoder.model.encoder.bert_model.embeddings.init_soft_tokens_w_given_label(
				label_tokens=label_tokens,
				smooth_alpha=smooth_alpha
			)
		elif self.config.param_type == "free_embeds":
			# Initialize label embedding parameters using start_label_idx
			crossencoder.model.encoder.bert_model.embeddings.init_label_embeds_from_word_embeds(label_tokens=label_tokens)
		else:
			raise NotImplementedError(f"self.config.param_type  = {self.config.param_type} not supported in init_w_given_label()")
			
			
	def get_optimizer(self, crossencoder):
		
		if self.config.param_type == "per_label_weight":
			# Optimize soft_label_weights parameters only, do not modify any other parameter
			type_optimization = "soft_label_weights"
		elif self.config.param_type == "per_pos_weights":
			# Optimize special label_token_embeds parameters only, do not modify word/position/type embedding parameters
			type_optimization = "soft_label_token_weights_per_pos"
		elif self.config.param_type == "free_embeds":
			# Optimize special label_token_embeds parameters only, do not modify word/position/type embedding parameters
			type_optimization = "label_token_embeds"
		else:
			raise NotImplementedError(f"self.config.param_type  = {self.config.param_type} not supported in init_w_given_label()")
			
		optimizer = get_bert_optimizer(
			models=[crossencoder],
			type_optimization=type_optimization,
			learning_rate=self.config.lr,
			weight_decay=0.0,
			optimizer_type=self.config.optimizer_type,
			verbose=False
		)
		
		return optimizer
		
		
	def get_parameter_entropy(self, crossencoder):
		"""
		Compute entropy of paramter distribution
		:param crossencoder:
		:return:
		"""
		try:
			if self.config.param_type == "per_label_weight":
				label_weights = crossencoder.model.encoder.bert_model.embeddings.get_label_weights()
				loss = torch.nn.CrossEntropyLoss()
				label_weights = label_weights.unsqueeze(0) # Change shape from (num_labels,) to (1, num_labels)
				entropy = loss(input=torch.log(label_weights), target=label_weights)
				return entropy
			# elif self.config.param_type == "per_pos_weights":
			# 	per_pos_weights = crossencoder.model.encoder.bert_model.embeddings.()
			else:
				raise NotImplementedError(f"self.config.param_type = {self.config.param_type} is not supported")
		except Exception as e:
			embed()
			raise e
		
	
	def run_gbi(self, ment_idxs):
		
		try:
			all_res = []
			for ment_idx in ment_idxs:
				curr_ment = self.mention_tokens_list[ment_idx]
				curr_res = self.run_gradient_based_inf_given_ment(
					mention=curr_ment,
					gt_label=self.gt_labels[ment_idx],
				)
				
				all_res += [curr_res]
			
			indices_scores_tokens_res = [list(zip(*x["explored_labels"])) for x in all_res]
			indices = [list(curr_indices) for _, curr_indices, curr_scores, _ in indices_scores_tokens_res]
			scores = [list(curr_scores) for _, curr_indices, curr_scores, _ in indices_scores_tokens_res]
			return indices, scores, all_res
		except Exception as e:
			embed()
			raise e
	
	
	def run_gradient_based_inf_given_ment(self, mention, gt_label):
		"""
		Run gradient-based inference starting from a particular entity
		:param mention:
		:param gt_label:
		:return:
		"""
		try:
			assert (not self.crossencoder.training), f"Crossencoder should be in eval mode"
			assert self.biencoder is None or (not self.biencoder.training), f"Biencoder should be in eval mode"
			
			tokenizer = self.crossencoder.tokenizer
			
			mention = mention if torch.is_tensor(mention) else torch.tensor(mention)
			mention = mention.to(self.crossencoder.device)
			
			all_label_tokens = self.complete_entity_tokens_list if torch.is_tensor(self.complete_entity_tokens_list) else torch.tensor(self.complete_entity_tokens_list)
			
			if self.config.init_method == "random":
				rng = np.random.default_rng(0)
				start_label_idx = rng.integers(0, self.num_labels)
			elif self.config.init_method == "gt":
				start_label_idx = gt_label
			else:
				raise NotImplementedError(f"self.config.init_method = {self.config.init_method} not supported")
			
			pred_label, explored_labels = self._run_gradient_based_inf_helper_w_cont_search(
				mention=mention,
				start_label_idx=start_label_idx,
				gt_label_idx=gt_label,
				use_gradients=True
			)
			pred_entity_titles = [f'{(x[0], x[1], "{:.4f}".format(x[2]),  " ".join(tokenizer.convert_ids_to_tokens(x[3])) )}'
									  for x in explored_labels]
			res = {
				"gt_label":int(gt_label),
				"explored_labels":explored_labels,
				"mention": ' '.join(tokenizer.convert_ids_to_tokens(mention.cpu().numpy().tolist())),
				"gt_entity": ' '.join(tokenizer.convert_ids_to_tokens(all_label_tokens[gt_label].cpu().numpy().tolist())),
				"pred_entity_titles":pred_entity_titles,
			}
			return res
		except Exception as e:
			embed()
			raise e
	
	
	def _run_gradient_based_inf_helper_w_cont_search(self, mention, start_label_idx, gt_label_idx, use_gradients):
		"""
		Run gradient-based inference by initializing using label at start_label_idx
		:param mention:
		:param start_label_idx:
		:param use_gradients: Boolean. If true, gradient-based inference is performed else not. Useful for debugging
		:return: curr_label_idx, explored_scores
			curr_label_idx : Label index of final label that inference converged to.
			explored_scores: List containing information about all labels explored during inference
		"""
		try:
			
			crossencoder = self.crossencoder
			all_label_tokens = self.complete_entity_tokens_list
			all_label_tokens = all_label_tokens if torch.is_tensor(all_label_tokens) else torch.tensor(all_label_tokens)
			
			self.init_w_given_label(
				crossencoder=crossencoder,
				label_tokens=all_label_tokens[start_label_idx][1:],
				label_idx=start_label_idx,
				smooth_alpha=self.config.smooth_alpha
			)
			
			# TODO: Also use a learning rate scheduler together with optimizer
			optimizer = self.get_optimizer(crossencoder=crossencoder)
			
			# Compute curr_pair representation using start_label_idx
			curr_label_idx = start_label_idx
			curr_label_tokens = all_label_tokens[curr_label_idx].cpu().numpy().tolist()[1:] # Remove cls token
			
			curr_pair = torch.cat((mention, torch.LongTensor(curr_label_tokens).to(device=mention.device))) # Shape : pair_rep_len
			curr_pair = curr_pair.unsqueeze(0).unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
			
			explored_scores = []
			
			
			# Perform gradient-based search for self.config.num_search_steps steps
			for search_iter in tqdm(range(self.config.num_search_steps), total=self.config.num_search_steps):
				
				assert curr_pair.shape[2] == self.max_ment_length + self.max_ent_length - 1, f" Ment-Ent pair len = {curr_pair.shape[2]} != Desired len = {self.max_ment_length} + {self.max_ent_length} - 1"
				
				# 1) Compute score wrt current entity/label. We want to maximize score so multiply with -1 and then minimize output
				score = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)
				explored_scores += [(search_iter, int(curr_label_idx), float(score.cpu().detach().numpy()[0][0]), curr_label_tokens)]
				
				loss = score
				
				if self.config.entropy_reg_alpha:
					entropy = self.get_parameter_entropy(
						crossencoder=crossencoder
					)
					loss = loss + ((search_iter%15)*self.config.entropy_reg_alpha)*entropy
					# loss = self.config.entropy_reg_alpha*entropy
					wandb.log({"entropy": entropy})
				
				wandb.log({"score": score, "loss": loss, "search_iter":search_iter})
				
				# 2) Clear out previous gradients if any, and compute gradients for this iteration
				if use_gradients:
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				
				with torch.no_grad():
					# # # Score after gradient update
					# score_after_update = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)

					# 3) Update current label
					# 3a) Extract updated label tokens representation after gradient update
					label_token_embeds  = crossencoder.model.encoder.bert_model.embeddings.get_label_token_embed()

					# 3b) Quantize (updated) label tokens representation to an actual entity
					curr_label_idx, curr_label_tokens = self._quantize_label_embeds(
						label_token_embeds=label_token_embeds,
						label_search_index=self.label_search_index,
						quant_method=self.config.quant_method,
						explored_nodes={x[0] for x in explored_scores}
					)
				
				if self.config.reinit_w_quant_interval > 0 and ((search_iter + 1) % self.config.reinit_w_quant_interval == 0):
					LOGGER.info(f"Re-initializing with  current label at search step = {search_iter}")
					self.init_w_given_label(
						crossencoder=crossencoder,
						label_tokens=curr_label_tokens,
						label_idx=curr_label_idx,
						smooth_alpha=self.config.smooth_alpha
					)
			
			
			# Score final set of parameters
			pred_score = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)
			explored_scores += [(self.config.num_search_steps, int(curr_label_idx), float(pred_score.cpu().detach().numpy()[0][0]), curr_label_tokens)]
			
			
			#### Now score discrete labels
			# Find top-k labels and score them
			with torch.no_grad():
				topk = 20
				explored_scores += self.score_topk_labels(
					mention=mention,
					crossencoder=crossencoder,
					topk=topk
				)
				
			# Re-Init w/ final label computed using quantization method and score it (without any label smoothening if applicable)
			self.init_w_given_label(
				crossencoder=crossencoder,
				label_tokens=curr_label_tokens,
				label_idx=curr_label_idx,
				smooth_alpha=1e-10
			)
			curr_pair = torch.cat((mention, torch.LongTensor(curr_label_tokens).to(device=mention.device))) # Shape : pair_rep_len
			curr_pair = curr_pair.unsqueeze(0).unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
			pred_score = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)
			explored_scores += [("final_quant", int(curr_label_idx), float(pred_score.cpu().detach().numpy()[0][0]), curr_label_tokens)]
			
			
			# Re-Init w/ gt label and score it (without any label smoothening, if applicable)
			self.init_w_given_label(
				crossencoder=crossencoder,
				label_tokens=self.complete_entity_tokens_list[gt_label_idx][1:].tolist(), # remove cls token
				label_idx=gt_label_idx,
				smooth_alpha=1e-10
			)
			curr_pair = torch.cat((mention, torch.LongTensor(self.complete_entity_tokens_list[gt_label_idx][1:]).to(device=mention.device))) # Shape : pair_rep_len
			curr_pair = curr_pair.unsqueeze(0).unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
			
			pred_score = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)
			explored_scores += [("gt_label", int(gt_label_idx), float(pred_score.cpu().detach().numpy()[0][0]), self.complete_entity_tokens_list[gt_label_idx].tolist())]
			
			return curr_label_idx, explored_scores
		except Exception as e:
			LOGGER.info("Error in _run_gradient_based_inf_helper_w_cont_search")
			embed()
			raise e
	
	
	def _run_gradient_based_inf_helper_w_discrete_search(self, crossencoder, token_embs, mention, start_label_idx, all_label_tokens, max_ment_length, lr, label_search_index):
		try:
			# TODO: Implement token-wise nearest nbr finding method and use that to find nearest nbr entity
			
			num_nbrs = 1 # TODO: Modify to support gradient-based search with a beam
			optimizer = get_bert_optimizer(models=[crossencoder], type_optimization="embeddings",
										   learning_rate=lr, weight_decay=0.0, optimizer_type="SGD")
			explored_labels = {}
			curr_label_idx = start_label_idx
			num_patience = 10
			for i in tqdm(range(len(all_label_tokens)), total=len(all_label_tokens)): # This is the max number of times we need to iterate.
				
				# Compute score wrt current entity/label
				curr_label = all_label_tokens[curr_label_idx].to(crossencoder.device)
				pair = torch.cat((mention, curr_label)) # Shape : pair_rep_len
				pair = pair.unsqueeze(0).unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
				
				score = crossencoder.score_candidate(pair, first_segment_end=max_ment_length)
				curr_label_rep = token_embs(curr_label)
				explored_labels[curr_label_idx] = score.cpu().detach().numpy()[0][0], len(torch.nonzero(curr_label))
				
				# Clear out previous gradients if any, and compute gradients for this iteration
				optimizer.zero_grad()
				score.backward()
				token_embs_grads = token_embs.weight.grad
			
				# Compute new entity/label representation
				# TODO: Use update function from some optimizer to maybe use other ideas like momentum etc
				
				for _patience_ctr in range(num_patience):
					next_label_rep = curr_label_rep + (_patience_ctr+1)*lr*token_embs_grads[curr_label]
					next_label_rep_1D = next_label_rep.view(1, -1).cpu().detach().numpy()
					
					# Find closest entity/label using index and repeat
					next_label_dist, next_label_idx = label_search_index.search(next_label_rep_1D, num_nbrs)
					
					assert num_nbrs == 1, f"num_nbrs = {num_nbrs}. We are just using first nearest nbr to num_nbrs should be 1, if not then change logic below"
					next_label_idx = next_label_idx[0][0]
					
					if next_label_idx not in explored_labels:
						break
				
				if next_label_idx not in explored_labels:
					curr_label_idx = next_label_idx
				else:
					break
					
				
			
			return explored_labels
		except Exception as e:
			embed()
			raise e
	
			
	def run_gradient_based_inf_dummy(self, crossencoder, mention, entity0, all_entities, max_ment_length):
		try:
			# Compute score for a given mention-gt entity pair - and perform gradient-based inference to find hard negatives
			
			lr = 0.01
			optimizer = get_bert_optimizer(models=[crossencoder], type_optimization="embeddings", learning_rate=lr,
										   weight_decay=0.0, optimizer_type="SGD")
			
			
			# assert isinstance(cross_encoder, CrossEncoderWrapper)
			# assert isinstance(cross_encoder.model.encoder, CrossBertWrapper)
			# assert isinstance(cross_encoder.model.encoder.bert_model, BertModel)
			
			m = crossencoder.model.encoder.bert_model
			LOGGER.info(f"Word Embeddings : {m.embeddings.word_embeddings}")
			LOGGER.info(f"Position Embeddings : {m.embeddings.position_embeddings}")
			LOGGER.info(f"Token Type Embeddings : {m.embeddings.token_type_embeddings}")
			
			token_embs = m.embeddings.word_embeddings
			# ent_token_embs = [token_embs(curr_ent_tokens.to(crossencoder.device)) for curr_ent_tokens in all_entities]
			
			
			if not torch.is_tensor(all_entities):
				all_entities = torch.tensor(all_entities)
			
			if not torch.is_tensor(mention):
				mention = torch.tensor(mention).to(crossencoder.device)
			
			entity0 = all_entities[0].to(crossencoder.device)
			ent0_token_embs_before = token_embs(entity0)
			
			pair = torch.cat((mention, entity0)) # Shape : pair_rep_len
			pair = pair.unsqueeze(0) # Shape : 1 x pair_rep_len
			pair = pair.unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
			
			pair = pair.to(crossencoder.device)
			LOGGER.info(f"Mention shape {mention.shape}")
			LOGGER.info(f"Entity shape {entity0.shape}")
			LOGGER.info(f"Pair shape {pair.shape}")
			
		
			score = crossencoder.score_candidate(pair, first_segment_end=max_ment_length)
	
			optimizer.zero_grad()
			score.backward()
			optimizer.step()
			
			ent0_token_embs_after = token_embs(entity0)
			
			LOGGER.info(f"Entity embedding before : {ent0_token_embs_before}")
			LOGGER.info(f"Entity embedding after : {ent0_token_embs_after}")
			
			# cross_encoder.model.encoder.bert_model.embeddings
			
			
			
			'''
			Next Steps:
			1) Get seq-of-embeddings for all entities, concatenate them to create a single entity embedding
			2) Index entity embeddings
			
			3) Repeat for finite number of steps:
			3a)		Compute mention-entity pair score
			3b) 	Compute gradients, and compute updated entity embeddings. We should NOT update actual token embeddings
					- just need to access gradients for those embeddings and manually compute updated entity representation
			3c) 	Find nearest nbr entity to current entity and go to 3a with the new entity
			
			4) Analysis
			4a) Look at entity descriptions are we hop from one entity to another
			4b) Do we get new entities in this process or are we usually stuck with just one entity -
				depends on how far apart these entities are.
			4b) See impact of learning rate on this.
			
			'''
			
			
			# 1) Figure out how to access embeddings for the model
			
			
			# 2) Figure out how to access gradients for those embeddings
			
			# 3) Get a new entity representation using previous embeddings and gradients ( at first it can be just simply adding gradients)
			# 3a) Think of other ways of computing a new representation
			
			# 4) Search over all possible entities - first need to index them somehow efficiently - look at set search repo
			
			
			embed()
		except Exception as e:
			embed()
			raise e
	
	
	@staticmethod
	def _get_single_label_rep_w_concat(label_token_embeds):
		"""
		Concat all label embeddings together to create a single label representation
		:param label_token_embeds:
		:return:
		"""
		# Concatenate all label embeddings into a single vector
		return label_token_embeds.view(1, -1).cpu().detach().numpy()
		
	
	@staticmethod
	def _get_single_label_rep_w_bienc(biencoder, label_token_embeds, label_token_idxs):
		"""
		Compute biencoder embedding and search for nearest nbr label
		:param biencoder:
		:param label_token_embeds:
		:param label_token_idxs:
		:return:
		"""

		# Instead of feeding in the label tokens, feed in the label rep to the biencoder model
		label_encoder = biencoder.model.label_encoder.bert_model
		assert isinstance(label_encoder, CustomLabelBert)
		
		# Switch to using custom_embeddings parameter in label_encoder
		label_encoder.embeddings = label_encoder.custom_embeddings

		# label_token_embeds come from label portion of input-label given to cross-encoder so append [cls] token embed to it
		cls_token_id  = torch.tensor(biencoder.tokenizer.cls_token_id).to(biencoder.device)
		cls_token_embed = label_encoder.embeddings.word_embeddings(cls_token_id).unsqueeze(0)
		label_token_embeds = torch.cat((cls_token_embed, label_token_embeds))

		# Initialize custom label embedding parameters using given label embeddings
		label_encoder.embeddings.init_label_embeds_from_given_embeds(label_token_embeds=label_token_embeds)
		
		# Encode the label into a single embedding
		label_token_idxs = label_token_idxs.to(biencoder.device)
		bienc_label_emb = biencoder.encode_label(label_token_idxs=label_token_idxs.unsqueeze(0))
		bienc_label_emb = bienc_label_emb.cpu().detach().numpy()
	
		# Switch back to using default_embeddings parameter in label_encoder
		label_encoder.embeddings = label_encoder.default_embeddings
		
		return bienc_label_emb
		
	
	@staticmethod
	def _viterbi_quantization(label_token_embeds, token_embeds, allowed_prev_vals, all_token_ids):
		"""
		
		:param label_token_embeds:
		:param token_embeds:
		:param allowed_prev_vals:
		:param all_token_ids:
		:return:
		"""
		try:
			per_pos_emission_prob = label_token_embeds @ (token_embeds.T)
			per_pos_emission_prob = per_pos_emission_prob.cpu().detach().numpy()
			
			max_seq_len = label_token_embeds.shape[0]
			
			emission_probs = {
				i:{token_idx: per_pos_emission_prob[i][token_idx] for token_idx in all_token_ids}
				for i in range(max_seq_len)
			}
			
			# LOGGER.info(f"Entity zero = {entity_tokens[0]}")
			# LOGGER.info("Discretizing soft sequence")
			final_max_score, max_score_seq = discretize_soft_sequence(
				allowed_prev_vals=allowed_prev_vals,
				emission_probs=emission_probs,
				seq_len=max_seq_len,
				val_vocab=all_token_ids
			)
			
			return max_score_seq
		except Exception as e:
			embed()
			raise e
	
	
	def _unigram_greedy_quantization(self, label_token_embeds, all_token_embeds, all_token_ids):
		"""
		
		:param label_token_embeds:
		:param all_token_embeds:
		:param all_token_ids:
		:return:
		"""
		
		if self.config.param_type == "free_embeds":
			max_seq_len = label_token_embeds.shape[0]
			per_pos_emission_prob = label_token_embeds @ (all_token_embeds.T)
			per_pos_emission_prob = per_pos_emission_prob.cpu().detach().numpy()
			
			emission_probs = {
				i:{token_idx: per_pos_emission_prob[i][token_idx] for token_idx in all_token_ids}
				for i in range(max_seq_len)
			}
			
		elif self.config.param_type == "per_pos_weights":
			emission_probs = self.crossencoder.model.encoder.bert_model.embeddings.get_per_pos_token_probs()
			max_seq_len = len(emission_probs)
		else:
			raise NotImplementedError(f"self.config.param_type == {self.config.param_type} not supported")

	
		final_max_score, max_score_seq = discretize_soft_sequence_wo_crf(
			emission_probs=emission_probs,
			seq_len=max_seq_len,
			val_vocab=all_token_ids
		)
		
		return final_max_score, max_score_seq
	
	
	def _label_greedy_quantization(self):
		"""
		
		:param label_token_embeds:
		:param token_embeds:
		:param allowed_prev_vals:
		:param all_token_ids:
		:return:
		"""
		try:
			if self.config.param_type == "per_label_weight":
				all_label_scores = self.crossencoder.model.encoder.bert_model.embeddings.get_label_weights()
			elif self.config.param_type == "per_pos_weights":
				emission_probs = self.crossencoder.model.encoder.bert_model.embeddings.get_per_pos_token_probs()
				max_seq_len = len(emission_probs)
				
				all_label_tokens = self.complete_entity_tokens_list[:, 1:]
				all_label_scores = []
				for curr_label_tokens in all_label_tokens:
					
					curr_score = 0
					for pos in range(max_seq_len):
						curr_score += np.log(emission_probs[pos][curr_label_tokens[pos]])
					
					
					all_label_scores += [curr_score]
				
				all_label_scores = torch.tensor(all_label_scores)
			else:
				raise NotImplementedError(f" self.config.param_type = {self.config.param_type} not implemented")
			
			
			max_score_label_idx  = int(torch.argmax(all_label_scores))
			max_score_label_tokens = self.complete_entity_tokens_list[max_score_label_idx][1:]
			
			return max_score_label_idx, max_score_label_tokens.tolist()
		except Exception as e:
			embed()
			raise e

	
	def _quantize_label_embeds(self, quant_method, label_token_embeds, label_search_index, explored_nodes):
		"""
		
		:param label_token_embeds:
		:param label_search_index: Index object for performing nearest nbr search for labels after embedding them appropriately
		:param quant_method:
		:param explored_nodes:
		:return:
			label_idx, label_tokens
			If label_idx = -1, then it indicates that label tokens may not correspond to an actual label
		"""
		if quant_method == "viterbi":
			max_score_seq = self._viterbi_quantization(
				label_token_embeds=label_token_embeds,
				token_embeds=self.all_token_embs,
				all_token_ids=self.all_token_ids,
				allowed_prev_vals=self.allowed_prev_vals
			)
			if tuple(max_score_seq) in self.token_seq_to_label_idx:
				pred_label_idx = self.token_seq_to_label_idx[tuple(max_score_seq)]
			else:
				pred_label_idx = -1
				
			return pred_label_idx, max_score_seq
		
		elif quant_method == "unigram_greedy":
			
			_, max_score_seq = self._unigram_greedy_quantization(
				label_token_embeds=label_token_embeds,
				all_token_embeds=self.all_token_embs,
				all_token_ids=self.all_token_ids,
			)
			
			return -1, max_score_seq
		elif quant_method == "label_greedy":
			pred_label_idx, max_score_seq = self._label_greedy_quantization()
			
			return pred_label_idx, max_score_seq
		elif quant_method in ["concat", "bienc"]:
			if quant_method == "concat":
				final_label_emb = self._get_single_label_rep_w_concat(label_token_embeds=label_token_embeds)
			# elif quant_method == "bienc":
			# 	final_label_emb = self._get_single_label_rep_w_bienc(
			# 		biencoder=biencoder,
			# 		label_token_embeds=label_token_embeds,
			# 		label_token_idxs=label_token_idxs
			# 	)
			else:
				raise NotImplementedError(f"Quantization method = {quant_method} not implemented")
			
			# Find closest entity/label using index.
			# Find 1 + len(explored_nodes) nbrs in case we stumble upon previously explored nodes
			pred_label_dist, pred_label_idx = label_search_index.search(final_label_emb, 1 + len(explored_nodes))
			pred_label_idx = pred_label_idx[0]
			
			# TODO: Add option for this
			# # Remove nearest nbrs present in explored_nodes
			# pred_label_idx = [idx for idx in pred_label_idx if idx not in explored_nodes]
			pred_label_idx = pred_label_idx[0]
	
			pred_label_tokens = self.complete_entity_tokens_list[pred_label_idx].tolist()[1:] # Remove cls token
			assert len(pred_label_tokens) == self.max_ent_length-1, f"len(pred_label_tokens) = {len(pred_label_tokens)} != self.self.max_ent_length-1 = {self.max_ent_length-1}"
			return pred_label_idx, pred_label_tokens
		else:
				raise NotImplementedError(f"Quantization method = {quant_method} not implemented")
	
	
	def get_topk_labels(self, crossencoder, topk):
		
		#TODO: This is a more general version of _quantize_label_embeds - Combine these two functions into a single one
		
		if self.config.quant_method == "label_greedy" and self.config.param_type == "per_label_weight":
			label_weights = crossencoder.model.encoder.bert_model.embeddings.get_label_weights()
			topk_scores, topk_label_idxs  = torch.topk(label_weights, k=topk)
			
			return topk_scores, topk_label_idxs, [self.complete_entity_tokens_list[idx][1:] for idx in topk_label_idxs]
		
		elif self.config.quant_method == "unigram_greedy" and self.config.param_type == "per_pos_weights":
			
			max_score, max_score_seq = self._unigram_greedy_quantization(
				label_token_embeds=None,
				all_token_embeds=self.all_token_embs,
				all_token_ids=self.all_token_ids,
			)
			
			# emission_probs = self.crossencoder.model.encoder.bert_model.embeddings.get_per_pos_token_probs()
			# max_seq_len = len(emission_probs)
			#
			# final_max_score, max_score_seq = discretize_soft_sequence_wo_crf(
			# 	emission_probs=emission_probs,
			# 	seq_len=max_seq_len,
			# 	val_vocab=self.all_token_ids
			# )
			#
			return [max_score], [-1], [max_score_seq]
		elif self.config.quant_method == "label_greedy" and self.config.param_type == "per_pos_weights":
			emission_probs = self.crossencoder.model.encoder.bert_model.embeddings.get_per_pos_token_probs()
			max_seq_len = len(emission_probs)
			
			all_label_tokens = self.complete_entity_tokens_list[:, 1:]
			all_label_scores = []
			for curr_label_tokens in all_label_tokens:
				
				curr_score = 0
				for pos in range(max_seq_len):
					curr_score += np.log(emission_probs[pos][curr_label_tokens[pos]])
				
				
				all_label_scores += [curr_score]
			
			all_label_scores = torch.tensor(all_label_scores)
			topk_scores, topk_label_idxs  = torch.topk(all_label_scores, k=topk)
			return topk_scores, topk_label_idxs, [self.complete_entity_tokens_list[idx][1:] for idx in topk_label_idxs]
		
		else:
			raise NotImplementedError(f'self.config.quant_method == {self.config.quant_method} and self.config.param_type == {self.config.param_type} not supported')
		
	
	def score_topk_labels(self, mention, crossencoder, topk):
		
		try:
			# FIXME: token_type and position embedding might be influenced by the initial entity
			explored_scores = []
			topk_scores, topk_label_idxs, topk_label_tokens = self.get_topk_labels(crossencoder=crossencoder, topk=topk)
			LOGGER.info(f"Top k scores  - \n{topk_scores}")
			
			for topk_iter, (curr_label_idx, curr_label_tokens) in enumerate(itertools.zip_longest(topk_label_idxs, topk_label_tokens)):
				
				curr_label_tokens = curr_label_tokens.tolist() if isinstance(curr_label_tokens, np.ndarray) else curr_label_tokens
				
				self.init_w_given_label(
					crossencoder=crossencoder,
					label_tokens=curr_label_tokens,
					label_idx=curr_label_idx,
					smooth_alpha=1e-10
				)
				
				curr_pair = torch.cat((mention, torch.LongTensor(curr_label_tokens).to(device=mention.device))) # Shape : pair_rep_len
				curr_pair = curr_pair.unsqueeze(0).unsqueeze(0) # Shape : 1 x 1 x pair_rep_len
	
				# Compute score with final label embeds and map final label embeddings to a label from label_index
				pred_score = -1*crossencoder.score_candidate(curr_pair, first_segment_end=self.max_ment_length)
				temp_score = topk_scores[topk_iter].cpu().numpy() if torch.is_tensor(topk_scores) else topk_scores[topk_iter]
				explored_scores += [( f"top-{topk_iter}-{temp_score:.4f}", int(curr_label_idx), float(pred_score.cpu().detach().numpy()[0][0]), curr_label_tokens)]
			
			return explored_scores
		except Exception as e:
			embed()
			raise e


def get_allowed_tokens_by_position(entity_tokens):

	n_ents, max_seq_len = entity_tokens.shape
	
	allowed_tokens = {i: sorted(list(set(entity_tokens[:, i].tolist()))) for i in range(max_seq_len)}
	
	return allowed_tokens


def get_token_pairs_by_position(entity_tokens):
	
	
	LOGGER.info("Creating entity token pairs by position ")
	n_ents, max_seq_len = entity_tokens.shape
	
	allowed_prev_vals = {}
	for i in tqdm(range(1, max_seq_len), total=max_seq_len):
		allowed_prev_vals[i] = defaultdict(list)
		for curr_ent_tokens in entity_tokens:
			
			allowed_prev_vals[i][curr_ent_tokens[i]] += [curr_ent_tokens[i-1]]
	
	LOGGER.info("Finished Creating entity token pairs by position ")
	
	return allowed_prev_vals


def discretize_soft_sequence(allowed_prev_vals, emission_probs, seq_len, val_vocab):
	"""
	
	:param allowed_prev_vals: Dict mapping position i to
									a dict which maps a token index to list of tokens compatible with it at position i-1
	:param emission_probs: Dict mapping position i to dict that stores score for each (allowed) token at that position
	:param seq_len: Length of sequence
	:param val_vocab: Global List of values allowed in the sequence
	:return:
	"""
	try:
		table = {}
		arg_max_table = {}
		
		# for i in tqdm(range(seq_len), total=seq_len):
		for i in range(seq_len):
			table[i] = {}
			arg_max_table[i] = {}
			for curr_val in val_vocab:
				
				# Skip values that are not allowed at position i i.e. which don't have a value in emission_probs[i] dictionary
				if curr_val not in emission_probs[i]: continue
				if i == 0:
					table[i][curr_val] = emission_probs[i][curr_val]
				else:
					# For-loop for debugging
					# for prev_val in allowed_prev_vals[i][curr_val]:
					# 	if prev_val not in table[i-1]: continue
					# 	scores = [table[i-1][prev_val] + emission_probs[i][curr_val]]
					
					# Skip prev_vals that don't have a score at table[i-1] position
					scores = [table[i-1][prev_val] + emission_probs[i][curr_val] for prev_val in allowed_prev_vals[i][curr_val] if prev_val in table[i-1]]
					
					if len(scores) == 0: continue # No allowed prev_val for curr_val at position i
					best_prev_val_idx = np.argmax(scores)
					best_prev_val = allowed_prev_vals[i][curr_val][best_prev_val_idx]
					best_score = scores[best_prev_val_idx]
					
					table[i][curr_val] = best_score
					
					arg_max_table[i][curr_val] = best_prev_val
	
		
		# Find best sequence score and then also find sequence corresponding to that score
		possible_vals = [j for j in val_vocab if j in table[seq_len-1]]
		final_max_score_token_idx = np.argmax([table[seq_len-1][j] for j in possible_vals])
		final_max_score_token = possible_vals[final_max_score_token_idx]
		final_max_score = table[seq_len-1][final_max_score_token]
		
	
		max_score_seq = [final_max_score_token]
		curr_max_score_token = final_max_score_token
		for i in range(seq_len-1, 0, -1):
			prev_max_score_token = arg_max_table[i][curr_max_score_token]
			
			# For position i-1
			max_score_seq += [prev_max_score_token]
			curr_max_score_token = prev_max_score_token
		
		max_score_seq.reverse()

		return final_max_score, max_score_seq
	except Exception as e:
		embed()
		raise e


def discretize_soft_sequence_wo_crf(emission_probs, seq_len, val_vocab):
	"""
	Find best value of each position in sequence independently
	:param emission_probs: Dict mapping position i to dict that stores score for each (allowed) token at that position
	:param seq_len: Length of sequence
	:param val_vocab: Global List of values allowed in the sequence
	:return:
	"""
	try:
		
		max_score_seq = []
		final_max_score = 0.
		for i in range(seq_len):
			scores = [emission_probs[i][curr_val] if curr_val in emission_probs[i] else -9999999999999 for curr_val in val_vocab]
		
			best_val_idx = np.argmax(scores)
			best_val = val_vocab[best_val_idx]
			best_score = scores[best_val_idx]
			max_score_seq += [best_val]
			final_max_score += np.log(best_score)
		
		return final_max_score, max_score_seq
	except Exception as e:
		embed()
		raise e


def debug_viterbi_quantization(label_embeds, token_embeds):

	try:
		from eval.debug_viterbi_decoding import discretize_soft_sequence, discretize_soft_sequence_wo_crf
		
		per_pos_emission_prob = label_embeds @ (token_embeds.T)
		per_pos_emission_prob = per_pos_emission_prob.cpu().detach().numpy()
		
		data_dir = "../../data/zeshel"
		domain = "lego"
		entity_tokens_file = f"{data_dir}/tokenized_entities/{domain}_128_bert_base_uncased.npy"
		
		# init tokenizer
		tokenizer = BertTokenizer.from_pretrained(
			"bert-base-uncased", do_lower_case=True
		)
		
		entity_tokens = np.load(entity_tokens_file)
		max_seq_len = label_embeds.shape[0]
		val_vocab = tokenizer.convert_tokens_to_ids(list(tokenizer.vocab.keys()))
		
		allowed_prev_vals = get_token_pairs_by_position(entity_tokens=entity_tokens)
		
	
		# emission_probs = {i:{v:1 for v in val_vocab} for i in range(max_seq_len)}
		#
		# # This should decode entity_0
		# emission_probs = {i:{entity_tokens[0][i]:1} for i in range(max_seq_len)}
		
		emission_probs = {
			i:{token_idx: per_pos_emission_prob[i][token_idx] for token_idx in val_vocab}
			for i in range(max_seq_len)
		}
		
		# LOGGER.info(f"Entity zero = {entity_tokens[0]}")
		LOGGER.info("Discretizing soft sequence")
		final_max_score, max_score_seq = discretize_soft_sequence(
			allowed_prev_vals=allowed_prev_vals,
			emission_probs=emission_probs,
			seq_len=max_seq_len,
			val_vocab=val_vocab
		)
	
		LOGGER.info(f"final_max_score = {final_max_score}")
		LOGGER.info(f"max_score_seq = {max_score_seq}")
		
		embed()
	except Exception as e:
		embed()
		raise e


def main():
	
	parser = argparse.ArgumentParser( description='Run gradient-based inference with a cross-encoder model')
	parser.add_argument("--config", type=str, required=True, help="config file")
	args, remaining_args = parser.parse_known_args()
	
	
	config = GradientBasedInfConfig(args.config)
	config.update_config_from_arg_list(dummy_config=GradientBasedInfConfig(), arg_list=remaining_args)

	Path(config.result_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present
	config.save_config(config.result_dir, "orig_config.json")
	
	config.validate_params()
	grad_inf_obj = GradientBasedInference(config=config)
	
	
	import wandb
	wandb.init(
		project=f"{config.exp_id}",
		dir=config.result_dir,
		config=config.__dict__
	)
	
	

	indices, scores, all_res = grad_inf_obj.run_gbi(
		ment_idxs=list(range(grad_inf_obj.config.n_ment))
	)

	final_res = {
		"pred_indices":indices,
		"pred_scores":scores,
		"all_res": all_res,
		"args": config.__dict__
	}
	try:
		with open(f"{config.result_dir}/preds.json", "w") as fout:
			json.dump(obj=final_res, fp=fout, indent=4)
	except Exception as e:
		embed()
		raise e
	
	
if __name__ == "__main__":
	main()
