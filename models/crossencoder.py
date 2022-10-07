import os
import sys
import copy
import torch
import wandb
import logging
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from IPython import embed
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
import pytorch_lightning as pl

from eval.eval_utils import score_topk_preds
from models.biencoder import BertWrapper
from models.params import ENT_TITLE_TAG, ENT_START_TAG, ENT_END_TAG
from utils.data_process import NSWDataset
from utils.config import Config
from utils.optimizer import get_bert_optimizer, get_scheduler

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def to_cross_bert_input(token_idxs, null_idx, first_segment_end):
	"""
	This function is used for preparing input for a cross-encoder bert model
	Create segment_idx and mask tensors for feeding the input to BERTs

	:param token_idxs: is a 2D int tensor.
	:param null_idx: idx of null element
	:param first_segment_end: idx where next segment i.e. label starts in the token_idxs tensor
	:return: return token_idx, segment_idx and mask
	"""
	# TODO: Verify that this is behaving as expected. Segment_idxs should be correct.
	segment_idxs = token_idxs * 0
	if first_segment_end > 0:
		segment_idxs[:, first_segment_end:] = token_idxs[:, first_segment_end:] > 0
	
	mask = token_idxs != null_idx
	# nullify elements in case self.NULL_IDX was not 0
	token_idxs = token_idxs * mask.long()
	
	return token_idxs, segment_idxs, mask


class CrossBertWEmbedsWrapper(nn.Module):
	"""
	Wrapper around BERT model which is used as a cross-encoder model.
	This first estimates contextualized embeddings for each input and then outputs score for the given input.
	"""
	def __init__(self, bert_model, pooling_type, bert_model_type='bert-base-uncased'):
		super(CrossBertWEmbedsWrapper, self).__init__()

		self.bert_model = bert_model
		
		self.pooling_type = pooling_type # TODO: Remove this param?
		self.bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
		
		# TODO: Remove hardcoded do_lower_case=True here
		tokenizer = BertTokenizer.from_pretrained(bert_model_type, do_lower_case=True)
		self.ENT_START_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_START_TAG)
		self.ENT_END_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_END_TAG)
		self.ENT_TITLE_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_TITLE_TAG)
		

	def forward(self, token_ids, segment_ids, attention_mask):
		input_1_embed, input_2_embed  = self.forward_for_embeds(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask
		)
		output_scores = torch.sum(input_1_embed*input_2_embed, dim=-1) # Shape: (batch_size, )
		
		# FIXME: Adding extra dim as wrapper around this class would call squeeze() before returning final scores, Remove this
		output_scores = output_scores.unsqueeze(1)
		return output_scores
	
	
	def forward_for_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return: returns two embedding tensors of shape (batch_size, embed_dim)
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		elif len(output) == 4:
			output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_start_tag_idxs = (token_ids == self.ENT_START_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_end_tag_idxs = (token_ids == self.ENT_END_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_title_idxs = (token_ids == self.ENT_TITLE_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_startm ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_start_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_start_tag_idxs.shape = {ent_start_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_end_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_end_tag_idxs.shape = {ent_end_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_title_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_title_idxs.shape = {ent_title_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_title_embeds = torch.stack([output_bert[i, ent_title_idxs[i, 1], :] for i in range(batch_size)])
		
		ent_start_embeds = torch.stack([output_bert[i, ent_start_tag_idxs[i, 1], :] for i in range(batch_size)])
		ent_end_embeds = torch.stack([output_bert[i, ent_end_tag_idxs[i, 1], :] for i in range(batch_size)])
		
		# LOGGER.info(f"ent_title_embeds.shape  = {ent_title_embeds.shape}")
		# LOGGER.info(f"ent_title_embeds_list.shape  = {ent_title_embeds_list[0].shape}")
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		input_1_embed = (ent_start_embeds + ent_end_embeds)/2 # shape: (batch_size, bert_output_dim)
		input_2_embed = ent_title_embeds # shape: (batch_size, bert_output_dim)
		
		return input_1_embed, input_2_embed
	
	
	def forward_for_input_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		Only extract embedding for input (eg mention in entity linking)
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return:
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_start_tag_idxs = (token_ids == self.ENT_START_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_end_tag_idxs = (token_ids == self.ENT_END_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_start, ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_start_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_start_tag_idxs.shape = {ent_start_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_end_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_end_tag_idxs.shape = {ent_end_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_start_embeds = torch.stack([output_bert[i, ent_start_tag_idxs[i, 1], :] for i in range(batch_size)])
		ent_end_embeds = torch.stack([output_bert[i, ent_end_tag_idxs[i, 1], :] for i in range(batch_size)])
		
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		input_embed = (ent_start_embeds + ent_end_embeds)/2 # shape: (batch_size, bert_output_dim)
		
		return input_embed
	
	
	def forward_for_label_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		Only extract embedding for label (eg entity in entity linking)
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return:
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		# elif len(output) == 4:
		# 	output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_title_idxs = (token_ids == self.ENT_TITLE_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_start, ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_title_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_title_idxs.shape = {ent_title_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_title_embeds = torch.stack([output_bert[i, ent_title_idxs[i, 1], :] for i in range(batch_size)])
		
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		label_embeds = ent_title_embeds # shape: (batch_size, bert_output_dim)
		
		return label_embeds
		
	
	def forward_per_layer(self, token_ids, segment_ids, attention_mask):
		input_1_embed_per_layer, input_2_embed_per_layer  = self.forward_for_embeds_per_layer(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask
		)
		output_scores_per_layer = torch.sum(input_1_embed_per_layer*input_2_embed_per_layer, dim=-1) # Shape: (batch_size, )???
		
		# FIXME: Adding extra dim as wrapper around this class would call squeeze() before returning final scores, Remove this
		output_scores_per_layer = output_scores_per_layer.unsqueeze(1)
		return output_scores_per_layer
	
	
	def forward_for_embeds_per_layer(self, token_ids, segment_ids, attention_mask):
		"""
		
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return: Two embedding tensors of shape (batch_size, num_layers, embed_dim)
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 4:
			output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_start_tag_idxs = (token_ids == self.ENT_START_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_end_tag_idxs = (token_ids == self.ENT_END_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_title_idxs = (token_ids == self.ENT_TITLE_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_startm ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_start_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_start_tag_idxs.shape = {ent_start_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_end_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_end_tag_idxs.shape = {ent_end_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_title_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_title_idxs.shape = {ent_title_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		final_input_1_embeds_list = []
		final_input_2_embeds_list = []
		for curr_layer_output in all_hidden_units[1:]: # Iterate over output of all layers. 0th layer is the input embedding layer so skipping it.
		
			# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
			# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
			ent_title_embeds = torch.stack([curr_layer_output[i, ent_title_idxs[i, 1], :] for i in range(batch_size)])
			
			ent_start_embeds = torch.stack([curr_layer_output[i, ent_start_tag_idxs[i, 1], :] for i in range(batch_size)])
			ent_end_embeds = torch.stack([curr_layer_output[i, ent_end_tag_idxs[i, 1], :] for i in range(batch_size)])
			
			# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
			input_1_embed = (ent_start_embeds + ent_end_embeds)/2 # shape: (batch_size, bert_output_dim)
			input_2_embed = ent_title_embeds # shape: (batch_size, bert_output_dim)
			
			final_input_1_embeds_list += [input_1_embed]
			final_input_2_embeds_list += [input_2_embed]
		
		
		final_input_1_embed = torch.stack(final_input_1_embeds_list, dim=1) # shape: (batch_size, bert_output_dim)
		final_input_2_embed = torch.stack(final_input_2_embeds_list, dim=1)
		
		return final_input_1_embed, final_input_2_embed
		
		

class CrossBertWrapper(BertWrapper):
	"""
	Wrapper around BERT model which is used as a cross-encoder model. This wrapper outputs scores for the given input.
	"""
	def __init__(self, bert_model, pooling_type):
		super(CrossBertWrapper, self).__init__(bert_model=bert_model,
											   output_dim=1,
											   pooling_type=pooling_type,
											   add_linear_layer=True)
		

	

	def forward(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		scores = super(CrossBertWrapper, self).forward(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			pooling_type=pooling_type
		)
		
		return scores


class CrossEncoderModule(torch.nn.Module):
	def __init__(self, bert_model, pooling_type, bert_args, cross_enc_type="default"):
		super(CrossEncoderModule, self).__init__()
		
		cross_bert = BertModel.from_pretrained(bert_model, **bert_args) # BERT Model for cross encoding input and labels
		
		self.bert_config = cross_bert.config
		
		if cross_enc_type == "default":
			self.encoder = CrossBertWrapper(
				bert_model=cross_bert,
				pooling_type=pooling_type
			)
		elif cross_enc_type == "w_embeds":
			self.encoder = CrossBertWEmbedsWrapper(
				bert_model=cross_bert,
				pooling_type=pooling_type,
				bert_model_type=bert_model
			)
		else:
			raise Exception(f"CrossEncoder type = {cross_enc_type} not supported")
		

	def forward(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		embedding = self.encoder(token_idx, segment_idx, mask)
		return embedding.squeeze(-1) # Remove last dim
	
	def forward_per_layer(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		embedding_per_layer = self.encoder.forward_per_layer(token_idx, segment_idx, mask)
		return embedding_per_layer.squeeze(-1) # Remove last dim
	
	def forward_for_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		embeddings_1, embeddings_2  = self.encoder.forward_for_embeds(token_idx, segment_idx, mask)
		return embeddings_1, embeddings_2
	
	def forward_for_input_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		if isinstance(self.encoder, CrossBertWEmbedsWrapper):
			return self.encoder.forward_for_input_embeds(token_idx, segment_idx, mask)
		elif isinstance(self.encoder, CrossBertWrapper):
			return self.encoder.forward_wo_linear(token_idx, segment_idx, mask)
		else:
			raise NotImplementedError(f"encoder of type={type(self.encoder)} not supported")
	
	def forward_for_label_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		if isinstance(self.encoder, CrossBertWEmbedsWrapper):
			return self.encoder.forward_for_label_embeds(token_idx, segment_idx, mask)
		elif isinstance(self.encoder, CrossBertWrapper):
			return self.encoder.forward_wo_linear(token_idx, segment_idx, mask)
		else:
			raise NotImplementedError(f"encoder of type={type(self.encoder)} not supported")
		
	

class CrossEncoderWrapper(pl.LightningModule):
	def __init__(self, config):
		super(CrossEncoderWrapper, self).__init__()
		assert isinstance(config, Config)
		
		# config.bert_args = {"output_attentions": True, "output_hidden_states":True} # Hack to allow for inference using all layers from BERT even when it was trained with just final layer representations
		self.config = config
		self.learning_rate = self.config.learning_rate
		
		# if os.path.isdir(self.config.result_dir):
		# 	LOGGER.addHandler(logging.FileHandler(f"{self.config.result_dir}/log_file.txt"))
		
		
		# init tokenizer
		self.tokenizer = BertTokenizer.from_pretrained(
			self.config.bert_model, do_lower_case=self.config.lowercase
		)
		self.NULL_IDX = self.tokenizer.pad_token_id
		
		# init model
		self.model = self.build_encoder_model()
		
		# Load parameters from file if it exists
		if os.path.exists(self.config.path_to_model):
			LOGGER.info(f"Loading parameters from {self.config.path_to_model}")
			self.update_encoder_model(skeleton_model=self.model, fname=self.config.path_to_model)
		else:
			LOGGER.info(f"Running with default parameters as self.config.path_to_model = {self.config.path_to_model} does not exist")

		# Move model to appropriate device ( No need with dataparallel)
		# self.device = self.config.device
		self.to(self.config.device)
		self.model = self.model.to(self.device)
		# if self.config.data_parallel:
		# 	self.model = torch.nn.DataParallel(self.model)
		self.save_hyperparameters()
		try:
			LOGGER.info(f"Model device is {self.device} {self.config.device}")
		except:
			pass
	
	@property
	def model_config(self):
		if isinstance(self.model, CrossEncoderModule):
			return self.model.bert_config
		elif isinstance(self.model, torch.nn.parallel.DataParallel) and isinstance(self.model.module, CrossEncoderModule):
			return self.model.module.bert_config
		else:
			raise Exception(f"model type = {type(self.model)} does not have a model config")
	
	@classmethod
	def load_model(cls, config):
		"""
		Load parameters from config file and create an object of this class
		:param config:
		:return:
		"""
		if isinstance(config, str):
			with open(config, "r") as f:
				return cls(Config(f))
		elif isinstance(config, Config):
			return cls(config)
		elif isinstance(config, dict):
			config_obj = Config()
			config_obj.__dict__.update(config)
			return cls(config_obj)
		else:
			raise Exception(f"Invalid config param = {config}")
	
	def save_model(self, res_dir):
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		
		self.config.encoder_wrapper_config = os.path.join(res_dir, "wrapper_config.json")
		self.save_encoder_model(res_dir=res_dir)
		self.config.save_config(res_dir=res_dir, filename="wrapper_config.json")
		
	def build_encoder_model(self):
		"""
		Build an (optionally pretrained) encoder model with the desired architecture.
		:return:
		"""
		cross_enc_type = self.config.cross_enc_type if hasattr(self.config, "cross_enc_type") else "default"
		
		bert_args = copy.deepcopy(self.config.bert_args)
		if hasattr(self.config, "use_all_layers") and self.config.use_all_layers:
			bert_args.update({"output_attentions":True, "output_hidden_states":True})
			
		return CrossEncoderModule(
			bert_model=self.config.bert_model,
			pooling_type=self.config.pooling_type,
			bert_args=bert_args,
			cross_enc_type=cross_enc_type
		)
	
	def save_encoder_model(self, res_dir):
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		
		model_file = os.path.join(res_dir, "model.torch")
		LOGGER.info("Saving encoder model to :{}".format(model_file))
		
		self.config.path_to_model = model_file
		if isinstance(self.model, torch.nn.DataParallel):
			torch.save(self.model.module.state_dict(), model_file)
		else:
			torch.save(self.model.state_dict(), model_file)
		
		
		model_config_file = os.path.join(res_dir, "model.config")
		self.model_config.to_json_file(model_config_file)
		
		self.tokenizer.save_vocabulary(res_dir)
	
	@staticmethod
	def update_encoder_model(skeleton_model, fname, cpu=False):
		"""
		Read state_dict in file fname and load parameters into skeleton_model
		:param skeleton_model: Model with the desired architecture
		:param fname:
		:param cpu:
		:return:
		"""
		if cpu:
			state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
		else:
			state_dict = torch.load(fname)
		if 'state_dict' in state_dict: # Load for pytorch lightning checkpoint
			model_state_dict = {}
			for key,val in state_dict['state_dict'].items():
				if key.startswith("model."):
					model_state_dict[key[6:]] = val
				else:
					model_state_dict[key] = val
					
			skeleton_model.load_state_dict(model_state_dict)
		else:
			skeleton_model.load_state_dict(state_dict)
	
	def encode(self, token_idxs, enc_to_use, first_segment_end):
		
		token_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=token_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		if enc_to_use == "input":
			return self.model.forward_for_input_embeds(
				token_idx=token_idxs,
				segment_idx=segment_idxs,
				mask=mask,
			)
		elif enc_to_use == "label":
			return self.model.forward_for_label_embeds(
				token_idx=token_idxs,
				segment_idx=segment_idxs,
				mask=mask,
			)
		else:
			raise NotImplementedError(f"Enc_to_use = {enc_to_use} not supported")
	
	def encode_input(self, input_token_idxs, first_segment_end=0):
		return self.encode(token_idxs=input_token_idxs, enc_to_use="input", first_segment_end=first_segment_end)

	def encode_label(self, label_token_idxs, first_segment_end=0):
		return self.encode(token_idxs=label_token_idxs, enc_to_use="label", first_segment_end=first_segment_end)
		

	# Score candidates given context input and label input
	def score_paired_input_and_labels(self, input_pair_idxs, first_segment_end):
		orig_shape = input_pair_idxs.shape
		input_pair_idxs = input_pair_idxs.view(-1, orig_shape[-1])
		
		# input_idxs.shape : batch_size x max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		
		# Score the pairs
		scores = self.model(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size,)

		scores = scores.view(orig_shape[:-1]) # Convert to shape compatible with original input shape
		return scores
		
		
	def score_candidate(self, input_pair_idxs, first_segment_end):
		return self.score_paired_input_and_labels(input_pair_idxs=input_pair_idxs, first_segment_end=first_segment_end)
	
	# Score candidates given context input and label input
	def score_paired_input_and_labels_per_layer(
		self,
		input_pair_idxs,
		first_segment_end
	):
		
		orig_shape = input_pair_idxs.shape
		input_pair_idxs = input_pair_idxs.view(-1, orig_shape[-1])
		
		# input_idxs.shape : batch_size, max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		
		# Score the pairs
		scores_per_layer = self.model.forward_per_layer(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size, num_layers )
		
		final_shape = orig_shape[:-1] + (scores_per_layer.shape[-1],) # Shape: (orig_shape, num_layers)
		scores_per_layer = scores_per_layer.view(final_shape) # Convert to shape compatible with original input shape
		# LOGGER.info(f"Testing per layer score computation : {orig_shape} -> {scores_per_layer.shape}")
		return scores_per_layer
		
	def score_candidate_per_layer(self, input_pair_idxs, first_segment_end):
		return self.score_paired_input_and_labels_per_layer(input_pair_idxs=input_pair_idxs, first_segment_end=first_segment_end)
	
	def embed_paired_input_and_labels(
		self,
		input_pair_idxs,
		first_segment_end
	):
		orig_shape = input_pair_idxs.shape
		input_pair_idxs = input_pair_idxs.view(-1, orig_shape[-1])
		
		# input_idxs.shape : batch_size x max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		
		# Get embeddings for inputs and labels -
		input_embeds, label_embeds = self.model.forward_for_embeds(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size, embed_dim)

		return input_embeds, label_embeds
	
	# FIXME: Remove this hack. Did this for model interpretability tool
	# def forward(self, input_pair_idxs):
	# 	return self.score_paired_input_and_labels(input_pair_idxs=input_pair_idxs, first_segment_end=128)
	
	def forward(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		loss, pos_scores, neg_scores = self.forward_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		return loss
		
	def forward_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		pos_scores = self.score_paired_input_and_labels(
			input_pair_idxs=pos_pair_idxs,
			first_segment_end=first_segment_end
		) # (batch_size, )
		
		batch_size, num_negs, seq_len = neg_pair_idxs.shape
		neg_scores = self.score_paired_input_and_labels(
			input_pair_idxs=neg_pair_idxs.view(batch_size*num_negs, seq_len),
			first_segment_end=first_segment_end
		) # (batch_size*num_negs, 1)
		
		neg_scores = neg_scores.view(batch_size, num_negs) # (batch_size, num_negs)
		
		loss = self.compute_loss_w_scores(
			pos_scores=pos_scores,
			neg_scores=neg_scores
		)
		return loss, pos_scores, neg_scores
	
	def forward_multi_label_w_scores(self, input_label_pair_idxs, batch_label_targets, first_segment_end):
		"""
		Score inputs in multi-label fashion. For each input, there could be more than one positive label
		as indicated by gt_label vector.
		This version returns both loss and score tensor
		:param input_label_pair_idxs: (batch_size, num_labels, seq_len) tensor containing input paired with num_labels labels
		:param batch_label_targets: (batch_size, num_labels) tensor containing 0/1 which indicate whether the label is positive or negative for the input that it is paired with
		:param first_segment_end:
		:return:
		"""
		
		batch_size, num_labels, seq_len = input_label_pair_idxs.shape
		scores = self.score_paired_input_and_labels(
			input_pair_idxs=input_label_pair_idxs.view(batch_size*num_labels, seq_len),
			first_segment_end=first_segment_end
		) # (batch_size*num_labels, 1)
		
		scores = scores.view(batch_size, num_labels) # (batch_size, num_labels)
		
		if self.config.loss_type == "hinge":
			loss_func = nn.MultiLabelMarginLoss()
			loss  = loss_func(scores, batch_label_targets)
		else:
			raise NotImplementedError(f"Loss function = {self.config.loss_type} not implemented")
		
		return loss, scores
	
	
	def forward_multi_label(self, input_label_pair_idxs, batch_label_targets, first_segment_end):
		"""
		Score inputs in multi-label fashion. For each input, there could be more than one positive label
		as indicated by gt_label vector.
		This version returns only loss
		:param input_label_pair_idxs: (batch_size, num_labels, seq_len) tensor containing input paired with num_labels labels
		:param batch_label_targets: (batch_size, num_labels) tensor containing 0/1 which indicate whether the label is positive or negative for the input that it is paired with
		:param first_segment_end:
		:return:
		"""
		loss, scores = self.forward_multi_label_w_scores(
			input_label_pair_idxs=input_label_pair_idxs,
			batch_label_targets=batch_label_targets,
			first_segment_end=first_segment_end
		)
		return loss
	
	
	def forward_per_layer(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		loss, pos_scores, neg_scores = self.forward_per_layer_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		return loss
	
		
	def forward_per_layer_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		pos_scores = self.score_paired_input_and_labels_per_layer(
			input_pair_idxs=pos_pair_idxs,
			first_segment_end=first_segment_end
		) # (batch_size, num_layers)
		
		batch_size, num_negs, seq_len = neg_pair_idxs.shape
		neg_scores = self.score_paired_input_and_labels_per_layer(
			input_pair_idxs=neg_pair_idxs.view(batch_size*num_negs, seq_len),
			first_segment_end=first_segment_end
		) # (batch_size*num_negs, num_layers)
		
		num_layers = neg_scores.shape[-1]
		neg_scores = neg_scores.view(batch_size, num_negs, num_layers) # (batch_size, num_negs, num_layers)
		
		# loss_list = []
		# for layer_iter in range(num_layers):
		# 	curr_loss = self.compute_loss_w_scores(
		# 		pos_scores=pos_scores[:,layer_iter],
		# 		neg_scores=neg_scores[:,:,layer_iter]
		# 	)
		# 	loss_list += [curr_loss]
		# debug_loss = 	torch.mean(torch.stack(loss_list))
		# LOGGER.info(f"Loss list :{debug_loss}")
		
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1, num_layers)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs, num_layers)
		
		# 0th col in each row in final_scores contained score for positive label
		target = torch.zeros((batch_size, num_layers), dtype=torch.long, device=final_scores.device)
		
		loss = F.cross_entropy(final_scores, target, reduction="mean")
	
		return loss, pos_scores, neg_scores
		
	
	def forward_as_bi_and_cross(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		loss, _, _ = self.forward_as_bi_and_cross_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		
		return loss
		
	def forward_as_bi_and_cross_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		"""
		Uses pos and neg pairs to compute loss as a cross-encoder, as a biencoder and combines the two losses
		using self.config.joint_train_alpha
		:param pos_pair_idxs:
		:param neg_pair_idxs:
		:param first_segment_end:
		:return:
		"""
		
		############################## Score pos and neg pairs using a cross-encoder model #############################
		cross_loss, pos_scores_w_cross, neg_scores_w_cross = self.forward_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		################################################################################################################
		
		
		################################################################################################################
		# Now split each pos and neg pair into input and label tokens, and score them in biencoder fashion
		################################################################################################################
		bi_loss, pos_scores_w_bi, neg_scores_w_bi = self.forward_as_biencoder_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		
		################################################################################################################
		
		assert 0 <= self.config.joint_train_alpha <= 1, f"self.config.joint_train_alpha = {self.config.joint_train_alpha} is < 0 or > 1"
		
		# Compute convex combination of two losses
		loss = self.config.joint_train_alpha*cross_loss + (1 - self.config.joint_train_alpha)*bi_loss
		
		return loss, (cross_loss, pos_scores_w_cross, neg_scores_w_cross), (bi_loss, pos_scores_w_bi, neg_scores_w_bi)
	
	
	def forward_as_bi_and_cross_mutual_distill(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		"""
		Uses pos and neg pairs to compute loss as a cross-encoder, as a biencoder and combines the two losses
		using self.config.joint_train_alpha.
		Additionally also computes loss for mutual distillation b/w biencoder and crossencoder
		
		:param pos_pair_idxs:
		:param neg_pair_idxs:
		:param first_segment_end:
		:return: Final loss combininig mutual distillation loss w/ loss from cross-encoder and bi-encoder
		"""
		
		final_loss, _, _, _ = self.forward_as_bi_and_cross_mutual_distill_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		
		return final_loss
	
	def forward_as_bi_and_cross_mutual_distill_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		"""
		Uses pos and neg pairs to compute loss as a cross-encoder, as a biencoder and combines the two losses
		using self.config.joint_train_alpha.
		Additionally also computes loss for mutual distillation b/w biencoder and crossencoder
		
		:param pos_pair_idxs:
		:param neg_pair_idxs:
		:param first_segment_end:
		:return:
		"""
		
		cross_and_bi_loss, (cross_loss, pos_scores_w_cross, neg_scores_w_cross), (bi_loss, pos_scores_w_bi, neg_scores_w_bi) = self.forward_as_bi_and_cross_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		
		# Compute loss for mutual distillation and combined with loss from cross-encoder and biencoder
		
		scores_w_cross = torch.cat((pos_scores_w_cross.unsqueeze(1), neg_scores_w_cross), dim=1) # (batch_size, 1 + num_negs)
		scores_w_bi = torch.cat((pos_scores_w_bi.unsqueeze(1), neg_scores_w_bi), dim=1) # (batch_size, 1 + num_negs)

		# Re-normalize scores for pos and neg labels for each input to sum up to 1
		torch_softmax = torch.nn.Softmax(dim=-1)
		scores_w_bi = torch_softmax(scores_w_bi)
	
		mutual_loss = F.cross_entropy(scores_w_cross, scores_w_bi, reduction="mean")
		
		assert 0 <= self.config.mutual_distill_alpha <= 1, f"self.config.mutual_distill_alpha = {self.config.mutual_distill_alpha} is < 0 or > 1"
		
		final_loss = self.config.mutual_distill_alpha*mutual_loss + (1 - self.config.mutual_distill_alpha)*cross_and_bi_loss
		return final_loss, cross_and_bi_loss, (cross_loss, pos_scores_w_cross, neg_scores_w_cross), (bi_loss, pos_scores_w_bi, neg_scores_w_bi)
	
	
	def forward_as_biencoder(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		"""
		Use cross-encoder parameter to embed each of input/mention and label separately like a dual/bi-encoder.
		:param pos_pair_idxs: Tensor of shape (batch_size, seq_len)
		:param neg_pair_idxs: Tensor of shape (batch_size, seq_len)
		:param first_segment_end: idx where next segment i.e. label starts in the token_idxs tensor
		:return: loss
		"""
		
		bi_loss, pos_scores_w_bi, neg_scores_w_bi = self.forward_as_biencoder_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		return bi_loss
	
	def forward_as_biencoder_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		"""
		Use cross-encoder parameter to embed each of input/mention and label separately like a dual/bi-encoder.
		:param pos_pair_idxs: Tensor of shape (batch_size, seq_len)
		:param neg_pair_idxs: Tensor of shape (batch_size, seq_len)
		:param first_segment_end: idx where next segment i.e. label starts in the token_idxs tensor
		:return: loss, pos_scores and neg_scores
		"""
		
		######################################## Split paired tokens into two parts ####################################
		input_idxs = pos_pair_idxs[:, :first_segment_end] # shape: (batch_size, num_input_tokens)
		pos_label_idxs = pos_pair_idxs[:, first_segment_end:] # shape: (batch_size, num_label_tokens)
		neg_label_idxs = neg_pair_idxs[:, :, first_segment_end:] # shape: (batch_size, num_negs, num_label_tokens)
		
		#############################  Append CLS tokens to label idxs here after splitting ############################
		cls_token = self.tokenizer.cls_token_id
		batch_size, num_negs, label_seq_len = neg_label_idxs.shape
		
		pos_cls = cls_token + torch.zeros((batch_size, 1)).to(pos_pair_idxs.device, pos_pair_idxs.dtype) # shape: (batch_size, 1)
		neg_cls = cls_token + torch.zeros((batch_size*num_negs, 1)).to(neg_pair_idxs.device, neg_pair_idxs.dtype) # shape: (batch_size*num_negs, 1)
		
		# (batch_size, num_label_tokens+1) <-- (batch_size, 1) <hstack> (batch_size, num_label_tokens)
		pos_label_idxs = torch.hstack((pos_cls, pos_label_idxs))
		
		# Reshape neg_label_idxs from  (batch_size, num_negs, num_label_tokens) to (batch_size*num_negs, num_label_tokens)
		neg_label_idxs = neg_label_idxs.view(batch_size*num_negs, label_seq_len) # shape : (batch_size*num_negs, num_label_tokens)
		
		# (batch_size*num_negs, num_label_tokens+1) <-- (batch_size*num_negs, 1) <hstack> (batch_size*num_negs, num_label_tokens)
		neg_label_idxs = torch.hstack((neg_cls, neg_label_idxs))
		
		######################################## COMPUTE INPUT AND LABEL EMBEDDINGS ####################################
		# Using first_segment_end=0 here as tokens given to biencoder consist of just one type of tokens
		#  - either just labels or or just input/context tokens but NEVER two different types of tokens/segments
		# concatenated together.
		input_embeds =  self.encode_input(input_token_idxs=input_idxs, first_segment_end=0) # shape: (batch_size, embed_dim)
		pos_label_embeds =  self.encode_label(label_token_idxs=pos_label_idxs, first_segment_end=0) # shape: (batch_size, embed_dim)
		neg_label_embeds =  self.encode_label(label_token_idxs=neg_label_idxs, first_segment_end=0) # shape: (batch_size*num_negs, embed_dim)
		
		embed_dim = neg_label_embeds.shape[1]
		neg_label_embeds = neg_label_embeds.view(batch_size, num_negs, embed_dim) # shape: (batch_size, num_negs, embed_dim)
		
		
		########################################### SCORE POS AND NEG LABELS ###########################################
		pos_scores_w_bi 	= torch.sum(input_embeds*pos_label_embeds, dim=1) # (batch_size, 1)
		
		# Add another dim to input_embs to score neg inputs for each input using matrix ops
		input_embeds = input_embeds.unsqueeze(1) # (batch_size, 1, embed_size)
		# input_embeds is broadcast along second dimension so that each input is multiplied with its negatives
		# (batch_size, num_negs, embed_size) =  (batch_size, num_negs, embed_size) x (batch_size, 1, embed_size)
		temp_prod 	= neg_label_embeds*input_embeds
		neg_scores_w_bi 	= torch.sum(temp_prod, dim=2) # (batch_size, num_negs)
		
		################################################## COMPUTE LOSS ################################################
		bi_loss  = self.compute_loss_w_scores(
			pos_scores=pos_scores_w_bi,
			neg_scores=neg_scores_w_bi
		)
		
		return bi_loss, pos_scores_w_bi, neg_scores_w_bi
	
	def forward_w_eval_metrics(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		pos_scores = self.score_paired_input_and_labels(pos_pair_idxs, first_segment_end) # (batch_size,)
		
		batch_size, num_negs, seq_len = neg_pair_idxs.shape
		neg_scores = self.score_paired_input_and_labels(neg_pair_idxs.view(batch_size*num_negs, seq_len),
														first_segment_end)
		
		neg_scores = neg_scores.view(batch_size, num_negs) # (batch_size, num_negs)
		
		loss = self.compute_loss_w_scores(pos_scores=pos_scores, neg_scores=neg_scores)
		
		res_metrics = self.compute_eval_metrics(pos_scores=pos_scores, neg_scores=neg_scores)
		res_metrics["loss"] = loss
		
		return res_metrics
	
	def forward_w_ranks(self, pos_pair_idxs, neg_pair_idxs, neg_pair_dists, first_segment_end):
		
		pos_scores = self.score_paired_input_and_labels(pos_pair_idxs, first_segment_end) # (batch_size, 1)
		
		batch_size, num_negs, seq_len = neg_pair_idxs.shape
		neg_scores = self.score_paired_input_and_labels(neg_pair_idxs.view(batch_size*num_negs, seq_len),
														first_segment_end)
		
		neg_scores = neg_scores.view(batch_size, num_negs) # (batch_size, num_negs)
		
		if self.config.loss_type == "rank_margin":
			loss = self.compute_margin_loss_w_ranks(pos_scores=pos_scores,
													neg_scores=neg_scores,
													neg_pair_dists=neg_pair_dists,
													margin=self.config.hinge_margin)
		elif self.config.loss_type == "rank_ce":
			loss = self.compute_cross_ent_loss_w_ranks(pos_scores=pos_scores,
													   neg_scores=neg_scores,
													   neg_pair_dists=neg_pair_dists,
													   dist_to_prob_method=self.config.dist_to_prob_method)
		else:
			raise NotImplementedError(f"Loss function of type = {self.config.loss_type} not implemented")
		
		return loss
	
	def forward_w_distill(self, pair_idxs, tgt_pair_scores, first_segment_end):
		torch_softmax = torch.nn.Softmax(dim=-1)
		
		batch_size, num_labels, seq_len = pair_idxs.shape
		pred_pair_scores = self.score_paired_input_and_labels(
			input_pair_idxs=pair_idxs.view(batch_size*num_labels, seq_len),
			first_segment_end=first_segment_end
		)
		
		pred_pair_scores = pred_pair_scores.view(batch_size, num_labels) # (batch_size, num_labels)
		
		# Now minimize loss b/w pred_pair_scores and tgt_pair_scores
		if self.config.loss_type == "ce":
			# Normalize tgt pairwise scores to a sum up to 1 for computing cross-entropy loss
			tgt_pair_scores = torch_softmax(tgt_pair_scores) # Shape : (batch_size, num_pairs)

			# Compute loss
			loss = F.cross_entropy(input=pred_pair_scores, target=tgt_pair_scores)
			return loss
		elif self.config.loss_type == "mse":
			loss = F.mse_loss(input=pred_pair_scores, target=tgt_pair_scores.to(pred_pair_scores.dtype))
			return loss
		# elif self.config.loss_type == "rank_margin":
		# 	raise NotImplementedError
		else:
			raise NotImplementedError(f"Loss = {self.config.loss_type} not supported")
		
	def forward_w_nsw_wo_batch(self, path_pos_pair_idxs, path_neg_pair_idxs, first_segment_end):
		
		try:
			"""
			TODO:
			(Edge) Cases to handle:
			1) Path length  = 1
			2) Multiple paths
			3) Add loss for ranking gt label over any other label
			"""
			path_pos_pair_idxs = torch.cat(path_pos_pair_idxs).to(self.device)
			path_pos_scores = self.score_paired_input_and_labels(path_pos_pair_idxs, first_segment_end) # (path_size, )
			path_pos_scores = path_pos_scores.unsqueeze(0) # (1, path_size)
			
			# path_neg_pair_idxs : Shape: path_size, <variable: number of nbrs of each node in path -1>
			temp_path_neg_pair_idxs = torch.cat(path_neg_pair_idxs, dim=1).to(self.device) # Shape: (1, total_negs_in_entire_path, seq_len)
			temp_path_neg_pair_idxs = temp_path_neg_pair_idxs.squeeze(0) # Shape: total_negs_in_entire_path, seq_len
			path_neg_scores = self.score_paired_input_and_labels(temp_path_neg_pair_idxs, first_segment_end) # Shape: (total_negs_in_entire_path, )
			path_neg_scores = path_neg_scores.unsqueeze(0) # Shape: (1, total_negs_in_entire_path)
			
			
			# Adding 1 because degree = num_negs + 1 (for positive edge)
			path_degree_info = [len(neg_pair_idxs)+1 for neg_pair_idxs in path_neg_pair_idxs]
			if self.config.loss_type == "ce":
				loss = torch.tensor(0., requires_grad=True).to(self.device)
				cum_path_degree_info = np.cumsum([0] + path_degree_info)
				for idx, curr_node_degree in enumerate(path_degree_info):
					pos_scores = path_pos_scores[:, idx]
					neg_scores = path_neg_scores[:, cum_path_degree_info[idx] : cum_path_degree_info[idx+1]]
					curr_loss = self.compute_cross_ent_loss(pos_scores=pos_scores,
															neg_scores=neg_scores)
					loss = loss + curr_loss
					
				loss = loss/len(path_degree_info) if len(path_degree_info) > 0 else loss
			else:
				raise NotImplementedError(f"Loss function of type = {self.config.loss_type} not implemented")
			
			return loss
		except Exception as e:
			embed()
			raise  e
	
	def compute_loss_w_scores(self, pos_scores, neg_scores):
		"""
		Compute various losses given scores for pos and neg labels
		:param pos_scores: Tensor of shape (batch_size, )
		:param neg_scores: Tensor of shape (batch_size, num_negs)
		:return:
		"""
		
		
		if self.config.loss_type == "bce":
			loss = self.compute_binary_cross_ent_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores
			)
			return loss
		elif self.config.loss_type == "ce":
			loss = self.compute_cross_ent_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores
			)
			return loss
		elif self.config.loss_type == "topk_ce":
			loss = self.compute_cross_ent_loss_w_topk(
				pos_scores=pos_scores,
				neg_scores=neg_scores,
				topk=self.config.topk_ce
			)
			return loss
		elif self.config.loss_type == "margin":
			loss = self.compute_margin_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores,
				margin=self.config.hinge_margin
			)
			return loss
		elif self.config.loss_type == "hinge" or self.config.loss_type == "hinge_sq":
			loss = self.compute_hinge_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores,
				hinge_margin=self.config.hinge_margin,
				squared=self.config.loss_type == "hinge_sq"
			)
			return loss
		elif self.config.loss_type == "rank_ce":
			batch_size, num_negs = neg_scores.shape
			loss = self.compute_cross_ent_loss_w_ranks(
				pos_scores=pos_scores,
				neg_scores=neg_scores,
				neg_pair_dists=torch.LongTensor([ np.arange(1, num_negs+1) for _ in range(batch_size)]).to(pos_scores.device),
				dist_to_prob_method=self.config.dist_to_prob_method
			)
			return loss
		else:
			raise NotImplementedError(f"Loss function of type = {self.config.loss_type} not implemented")
	
	@staticmethod
	def convert_dist_to_probs(dist_mat, method):
		"""
		Convert each row containing distances to probability distribution
		:param dist_mat: (batch_size, 1 + num_negs) shape matrix. Each row needs to be normalized into a probability distribution
		:param method: Method to use when converting distances to probability distributions
		:return: Matrix where each row corresponds to normalized probability distribution
		"""
	
		if method.startswith("negate"):
			max_vals = torch.max(dist_mat, dim=1)[0].unsqueeze(1) # Shape : (batch_size, 1)
			sim_mat =  max_vals - dist_mat
			sim_mat = sim_mat.to(dtype=torch.float)
		elif method.startswith("reciprocal"):
			sim_mat = 1/(1 + dist_mat)
		else:
			raise Exception(f"Method = {method} not supported in convert_dist_to_probs()")
		
		if method.endswith("softmax"):
			final_sim_mat = torch.nn.functional.softmax(sim_mat, dim=-1)
		elif method.endswith("linear"):
			final_sim_mat = torch.nn.functional.normalize(sim_mat, dim=-1, p=1)
		else:
			raise Exception(f"Method = {method} not supported in convert_dist_to_probs()")
		
		return final_sim_mat
		
	@staticmethod
	def compute_eval_metrics(pos_scores, neg_scores):
		"""
		
		:param pos_scores: score tensor of shape (batch_size,)
		:param neg_scores: score tensor of shape (batch_size, num_negs)
		:return:
		"""
		batch_size, num_negs = neg_scores.shape
		
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row in final_scores contained score for positive label
		target = np.zeros(batch_size)
		
		# Sort scores based on preds
		topk_scores, topk_indices = final_scores.topk(k=num_negs+1)
		topk_preds = {"indices":topk_indices.cpu().detach().numpy(),
					  "scores":topk_scores.cpu().detach().numpy()}
		
		res_metrics = score_topk_preds(gt_labels=target, topk_preds=topk_preds)
		
		return res_metrics
	
	@staticmethod
	def compute_binary_cross_ent_loss(pos_scores, neg_scores):
		"""
		Compute binary cross-entropy loss for each pos and neg score
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:return:
		"""
		pos_target = torch.ones(pos_scores.shape, device=pos_scores.device)
		neg_target = torch.zeros(neg_scores.shape, device=neg_scores.device)
		
		pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_target, reduction="mean")
		neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_target, reduction="mean")
		
		loss = (pos_loss + neg_loss)/2
		
		return loss
	
	@staticmethod
	def compute_cross_ent_loss(pos_scores, neg_scores):
		"""
		Compute cross-entropy loss
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:return:
		"""
		
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row in final_scores contained score for positive label
		target = torch.zeros((batch_size), dtype=torch.long, device=final_scores.device)
		
		loss = F.cross_entropy(final_scores, target, reduction="mean")
		return loss
		
	@staticmethod
	def compute_cross_ent_loss_w_ranks(pos_scores, neg_scores, neg_pair_dists, dist_to_prob_method):
		"""
		Compute cross-entropy loss using a target probability distribution computed using ranks (distances) of negative examples
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:param neg_pair_dists: (batch_size, num_neg) dim tensor. Each row stores distance values assigned to corresponding negative example
		:param dist_to_prob_method: Method to use for converting distances to probabilities
		:return:
		"""
		batch_size, num_negs = neg_scores.shape
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# This method normalizes score for pos and neg labels together
		# # 0th col in each row in final_scores contained score for positive label so 0th col in final_dists should be zero as pos label has zero distance from itself.
		# zeros = torch.zeros((batch_size,1), dtype=torch.long, device=neg_scores.device)
		# final_dists = torch.cat((zeros, neg_pair_dists), dim=1) # (batch_size, 1 + num_negs)
		# assert final_dists.shape == (batch_size, 1 + num_negs), f"Final dist shape = {final_dists.shape} does not match {batch_size, 1+num_negs}"
		#
		# target_probs = CrossEncoderWrapper.convert_dist_to_probs(
		# 	dist_mat=final_dists,
		# 	method=dist_to_prob_method
		# )
		
		
		# First normalize scores for neg labels and then renormalize score of pos and neg labels combined
		# shape: (batch_size, num_negs)
		neg_target_probs = CrossEncoderWrapper.convert_dist_to_probs(
			dist_mat=neg_pair_dists,
			method=dist_to_prob_method
		)
		# 0th col in each row in final_scores contained score for positive label so 0th col in final_dists should be zero as pos label has zero distance from itself.
		ones = torch.ones((batch_size,1), dtype=torch.long, device=neg_scores.device)
		target_probs = torch.cat((ones, neg_target_probs), dim=1) # (batch_size, 1 + num_negs)
		
		# Re-normalize each scores for pos and neg labels for each input to sum up to 1
		target_probs = torch.nn.functional.normalize(target_probs, dim=-1, p=1)
	
		loss = F.cross_entropy(final_scores, target_probs, reduction="mean")
		return loss
	
	@staticmethod
	def compute_cross_ent_loss_w_topk(pos_scores, neg_scores, topk):
		"""
		Compute cross-entropy loss
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:return:
		"""
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores
		
		# Find and use just top-k neg scores if num_negs is greater than topk
		if neg_scores.shape[1] > topk:
			neg_scores, _ = torch.topk(neg_scores, k=topk)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row contained score for positive label
		target = torch.zeros((batch_size), dtype=torch.long, device=final_scores.device)
		
		loss = F.cross_entropy(final_scores, target, reduction="mean")
		
		return loss
	
	@staticmethod
	def compute_hinge_loss(pos_scores, neg_scores, hinge_margin, squared):
		"""
		Compute (squared) hinge loss with given pos and neg scores
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:param hinge_margin: margin to use for hinge loss computation
		:param squared: Flag to indicate if squared hinge loss is used or not.
		:return:
		"""
		assert hinge_margin >= 0
		
		pos_scores[pos_scores > hinge_margin] = 0.
		neg_scores[neg_scores < -hinge_margin] = 0.
		
		if squared:
			pos_scores = hinge_margin - pos_scores
			neg_scores = hinge_margin + neg_scores
			
			pos_loss = torch.mean(pos_scores*pos_scores)
			neg_loss = torch.mean(neg_scores*neg_scores)
		else:
			pos_loss = -torch.mean(pos_scores)
			neg_loss = torch.mean(neg_scores)
		
		loss = (pos_loss + neg_loss)/2
		
		return loss
	
	@staticmethod
	def compute_margin_loss(pos_scores, neg_scores, margin):
		"""
		Compute (squared) hinge loss with given pos and neg scores
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:param margin: margin to use for hinge loss computation
		:return:
		"""
		assert margin >= 0
		
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row in final_scores contained score for positive label
		target = torch.zeros((batch_size), dtype=torch.long, device=final_scores.device)
		
		loss = F.multi_margin_loss(final_scores, target, margin=margin, reduction="mean")
		
		return loss
	
	@staticmethod
	def compute_margin_loss_w_ranks(pos_scores, neg_scores, neg_pair_dists, margin):
		"""
		Compute (squared) hinge loss with given pos and neg scores
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:param margin: margin to use for hinge loss computation
		:return:
		"""
		
		assert margin >= 0
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		# Ensure effective margin = margin*dist b/w positive and corresponding negatives
		loss_w_margin = pos_scores - neg_scores + margin*neg_pair_dists
		loss_w_margin[loss_w_margin < 0] = 0 # Ignore if difference is less than zero
	
		loss = torch.mean(loss_w_margin)
		return loss
		
	def configure_optimizers(self):
		optimizer = get_bert_optimizer(
			models=[self],
			type_optimization=self.config.type_optimization,
			learning_rate=self.learning_rate,
			weight_decay=self.config.weight_decay,
			optimizer_type="AdamW"
		)
		# len_data = len(self.trainer._data_connector._train_dataloader_source.instance)
		len_data = self.trainer.datamodule.train_data_len
		# len_data is already adjusted taking into batch_size and grad_acc_steps into account so pass 1 for these
		scheduler = get_scheduler(
			optimizer=optimizer,
			epochs=self.config.num_epochs,
			warmup_proportion=self.config.warmup_proportion,
			len_data=len_data,
			batch_size=1,
			grad_acc_steps=1
		)
		
		lr_scheduler_config = {
			"scheduler": scheduler,
			"interval": "step",
			"frequency": 1,
		}
		return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}
	
	def training_step(self, train_batch, batch_idx):
		
		if len(train_batch) == 2 and self.config.data_type == "xmc" and self.config.pos_strategy == "keep_pos_together":
			batch_input_label_pair_idxs, batch_label_targets = train_batch
			loss = self.forward_multi_label(
				input_label_pair_idxs=batch_input_label_pair_idxs,
				batch_label_targets=batch_label_targets,
				first_segment_end=self.config.max_input_len
			)
			return loss
		elif len(train_batch) == 2 and self.config.neg_strategy not in ["bienc_distill"]:
			batch_pos_pairs, batch_neg_pairs = train_batch
			if (not self.config.use_all_layers) and self.config.joint_train_alpha == 1 and self.config.mutual_distill_alpha == 0:
				return self.forward(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
			elif (not self.config.use_all_layers) and (0 < self.config.joint_train_alpha < 1) and (self.config.mutual_distill_alpha == 0):
				return self.forward_as_bi_and_cross(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
			elif (not self.config.use_all_layers) and (0 <= self.config.joint_train_alpha <= 1) and (0 < self.config.mutual_distill_alpha < 1):
				return self.forward_as_bi_and_cross_mutual_distill(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
			elif self.config.use_all_layers and self.config.joint_train_alpha == 1 and self.config.mutual_distill_alpha == 0:
				return self.forward_per_layer(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
			else:
				raise NotImplementedError(f"Following combination of values not supported"
										  f"self.config.use_all_layers = {self.config.use_all_layers} "
										  f"self.config.joint_train_alpha = {self.config.joint_train_alpha} "
										  f"self.config.mutual_distill_alpha = {self.config.mutual_distill_alpha}")
			
		elif len(train_batch) == 2 and self.config.neg_strategy in ["bienc_distill"]:
			batch_pairs, batch_pair_scores  = train_batch
			loss = self.forward_w_distill(
				pair_idxs=batch_pairs,
				tgt_pair_scores=batch_pair_scores,
				first_segment_end=self.config.max_input_len
			)
			return loss
		elif len(train_batch) == 3 and self.config.strategy in ["nsw_graph_rank", "bienc_hard_negs_w_knn_rank", "tfidf_hard_negs_w_knn_rank"]:
			batch_pos_pairs, batch_neg_pairs, batch_neg_dists = train_batch
			loss = self.forward_w_ranks(
				pos_pair_idxs=batch_pos_pairs,
				neg_pair_idxs=batch_neg_pairs,
				neg_pair_dists=batch_neg_dists,
				first_segment_end=self.config.max_input_len
			)
			return loss
		else:
			raise NotImplementedError(f"Number of elements in train_batch = {len(train_batch)} is not supported")
		
	def validation_step(self, val_batch, batch_idx):
		
		if len(val_batch) == 2 and self.config.data_type == "xmc" and self.config.pos_strategy == "keep_pos_together":
			batch_input_label_pair_idxs, batch_label_targets = val_batch
			loss = self.forward_multi_label(
				input_label_pair_idxs=batch_input_label_pair_idxs,
				batch_label_targets=batch_label_targets,
				first_segment_end=self.config.max_input_len
			)
			assert self.config.ckpt_metric == "loss", f"Checkpoint metric = {self.config.ckpt_metric} not supported."
			return {"loss": loss}
		elif len(val_batch) == 2 and self.config.neg_strategy not in ["bienc_distill"]:
			batch_pos_pairs, batch_neg_pairs = val_batch
			res_metrics = {}
			if (not self.config.use_all_layers) and self.config.joint_train_alpha == 1 and self.config.mutual_distill_alpha == 0:
				loss, pos_scores, neg_scores = self.forward_w_scores(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
				res_metrics["loss"] = loss
			elif (not self.config.use_all_layers) and (0 < self.config.joint_train_alpha < 1) and (self.config.mutual_distill_alpha == 0):
				loss, cross_data, bi_data = self.forward_as_bi_and_cross_w_scores(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
				loss_cross, pos_scores, neg_scores = cross_data
				loss_bi, _, _ = bi_data
				res_metrics["loss"] = loss
				res_metrics["loss_cross"] = loss_cross
				res_metrics["loss_bi"] = loss_bi
			elif (not self.config.use_all_layers) and (0 <= self.config.joint_train_alpha <= 1) and (0 < self.config.mutual_distill_alpha < 1):
				final_loss, cross_and_bi_loss, cross_data, bi_data = self.forward_as_bi_and_cross_mutual_distill_w_scores(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
				loss_cross, pos_scores, neg_scores = cross_data
				loss_bi, _, _ = bi_data
				res_metrics["loss"] = final_loss
				res_metrics["loss_cross"] = loss_cross
				res_metrics["loss_bi"] = loss_bi
				
			elif self.config.use_all_layers and self.config.joint_train_alpha == 1:
				loss = self.forward_per_layer(
					pos_pair_idxs=batch_pos_pairs,
					neg_pair_idxs=batch_neg_pairs,
					first_segment_end=self.config.max_input_len
				)
				res_metrics["loss"] = loss
				return res_metrics
			else:
				raise NotImplementedError(f"self.config.joint_train_alpha = {self.config.joint_train_alpha} > 1 or < 0 not supported ")
			
			if self.config.ckpt_metric == "mrr":
				temp_metrics = self.compute_eval_metrics(
					pos_scores=pos_scores,
					neg_scores=neg_scores,
				)
				res_metrics.update(temp_metrics)
				return res_metrics
			elif self.config.ckpt_metric == "loss":
				return res_metrics
			else:
				raise NotImplementedError(f"ckpt metric = {self.config.ckpt_metric} not supported")
			
			
		elif len(val_batch) == 2 and self.config.neg_strategy in ["bienc_distill"]:
			assert self.config.ckpt_metric == "loss", f"ckpt metric = {self.config.ckpt_metric} not supported"
			batch_pairs, batch_pair_scores  = val_batch
			loss = self.forward_w_distill(
				pair_idxs=batch_pairs,
				tgt_pair_scores=batch_pair_scores,
				first_segment_end=self.config.max_input_len
			)
			return {"loss": loss}
		elif len(val_batch) == 3:
			assert self.config.ckpt_metric == "loss", f"ckpt metric == {self.config.ckpt_metric} not supported"
			batch_pos_pairs, batch_neg_pairs, batch_neg_dists = val_batch
			loss = self.forward_w_ranks(
				pos_pair_idxs=batch_pos_pairs,
				neg_pair_idxs=batch_neg_pairs,
				neg_pair_dists=batch_neg_dists,
				first_segment_end=self.config.max_input_len
			)
			return {"loss": loss}
		else:
			raise NotImplementedError(f"Number of elements in val_batch = {len(val_batch)} is not supported")
		
	def validation_epoch_end(self, outputs):
		
		super(CrossEncoderWrapper, self).validation_epoch_end(outputs=outputs)
		
		eval_loss = torch.mean(torch.tensor([scores["loss"] for scores in outputs])) # Avg loss numbers
		
		if self.config.ckpt_metric == "mrr":
			eval_metric = np.mean([float(scores["mrr"]) for scores in outputs]) # Avg MRR numbers
			# Usually we use loss for eval_metric and want to find params that minimize this loss.
			# Since higher MRR is better, we multiply eval_metric with -1 so that we can still use min of this metric for checkpointing purposes
			eval_metric = -1*eval_metric
			
			self.log(f"dev_loss", eval_loss, sync_dist=True, on_epoch=True, logger=True)
			
		elif self.config.ckpt_metric == "loss":
			eval_metric = eval_loss
		else:
			raise NotImplementedError(f"ckpt metric = {self.config.ckpt_metric} not supported")
		
		self.log(f"dev_{self.config.ckpt_metric}", eval_metric, sync_dist=True, on_epoch=True, logger=True)
		
		if self.config.joint_train_alpha != 1.0:
			eval_loss_bi = torch.mean(torch.tensor([scores["loss_bi"] for scores in outputs])) # Avg loss numbers
			eval_loss_cross = torch.mean(torch.tensor([scores["loss_cross"] for scores in outputs])) # Avg loss numbers
			
			self.log(f"dev_loss_bi", eval_loss_bi, sync_dist=True, on_epoch=True, logger=True)
			self.log(f"dev_loss_cross", eval_loss_cross, sync_dist=True, on_epoch=True, logger=True)
			
	def on_train_start(self):
		super(CrossEncoderWrapper, self).on_train_start()
		LOGGER.info("On Train Start")
		self.log("train_step", self.global_step, logger=True)
	
	def on_train_epoch_start(self):
		
		super(CrossEncoderWrapper, self).on_train_epoch_start()
		if self.config.reload_dataloaders_every_n_epochs and (self.current_epoch % self.config.reload_dataloaders_every_n_epochs == 0)\
				and self.trainer.checkpoint_callback:
			LOGGER.info(f"\n\n\t\tResetting model checkpoint callback params in epoch = {self.current_epoch}\n\n")
			for checkpoint_callback in self.trainer.checkpoint_callbacks:
				checkpoint_callback.current_score = None
				checkpoint_callback.best_k_models = {}
				checkpoint_callback.kth_best_model_path = ""
				checkpoint_callback.best_model_score = None
				checkpoint_callback.best_model_path = ""
				checkpoint_callback.last_model_path = ""
			
	def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
		super(CrossEncoderWrapper, self).on_train_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx, unused=unused)
		
		if (self.global_step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
			# train_loss = float(outputs['loss'].cpu().numpy())
			# wandb.log({"train/epoch": self.current_epoch,
			# 		   "train/step": self.global_step,
			# 		   "train/loss": train_loss})
			
			self.log("train_loss", outputs, on_epoch=True, logger=True)
			self.log("train_step", self.global_step, logger=True)
			self.log("train_epoch", self.current_epoch, logger=True)
