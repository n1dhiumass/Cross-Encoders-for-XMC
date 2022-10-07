import os
import json
import torch
import random
import argparse
import warnings
import numpy as np

from pathlib import Path


class BaseConfig(object):
	
	def __init__(self, filename=None):
		
		self.config_name = filename
		self.seed 			= 1234
		
	def to_json(self):
		return json.dumps(filter_json(self.__dict__), indent=4, sort_keys=True)

	def save_config(self, res_dir, filename='config.json'):
		fname = os.path.join(res_dir, filename)
		with open(fname, 'w') as fout:
			json.dump(filter_json(self.__dict__),fout, indent=4, sort_keys=True)
		return fname
	
	def __getstate__(self):
		state = dict(self.__dict__)
		if "logger" in state:
			del state['logger']
			
		return state
	
	def update_random_seeds(self, seed):
		raise NotImplementedError
	
	@staticmethod
	def get_parser_for_args(dummy_config):
		
		parser = argparse.ArgumentParser(description='Get config from str')
		
		################################## OPTIONAL ARGUMENTS TO OVERWRITE CONFIG FILE ARGS#############################
		for config_arg in dummy_config.__dict__:
			def_val = dummy_config.__getattribute__(config_arg)
			if type(def_val) == tuple:
				def_val = def_val[0]
				
			arg_type = type(def_val) if def_val is not None else str
			arg_type = arg_type if arg_type is not dict else str
			
			if arg_type == list or arg_type == tuple:
				arg_type = type(def_val[0]) if len(def_val) > 0 else str
				parser.add_argument('--{}'.format(config_arg), nargs='+', type=arg_type, default=None,
									help='If not specified then value from config file will be used')
			else:
				parser.add_argument('--{}'.format(config_arg), type=arg_type, default=None,
									help='If not specified then value from config file will be used')
		################################################################################################################
	
		return parser
	
	def update_config_from_arg_list(self, dummy_config, arg_list):
		
		parser = BaseConfig.get_parser_for_args(dummy_config=dummy_config)
		print(f"Parsing arg_list = {arg_list}\n\n")
		args = parser.parse_args(arg_list)
		
		for config_arg in self.__dict__:
			def_val = getattr(args, config_arg)
			if def_val is not None:
				
				old_val = self.__dict__[config_arg]
				self.__dict__.update({config_arg: def_val})
				new_val = self.__dict__[config_arg]
				print("Updating Config.{} from {} to {} using arg_val={}".format(config_arg, old_val, new_val, def_val))
		
		self.update_random_seeds(self.seed)

	
class Config(BaseConfig):
	def __init__(self, filename=None):

		super(Config, self).__init__(filename=filename)
		self.config_name 	= filename

		self.save_code		= True
		self.base_res_dir	= "../../results"
		self.exp_id			= ""
		self.res_dir_prefix	= "" # Prefix to add to result dir name
		self.misc			= ""

		self.seed 			= 1234
		self.n_procs		= 20

		self.max_time = "06:23:55:00" # 7 days - 5 minutes
		self.fast_dev_run = 0 # Run a few batches from train/dev for sanity check

		self.print_interval = 10
		self.eval_interval  = 800.0


		# Data specific params
		self.data_type 		= "dummy"
		self.data_dir		= "None"
		self.trn_files   = {"dummy_domain":("dummy_ment_file", "dummy_ent_file", "dummy_ent_tokens_file")}
		self.dev_files   = {"dummy_domain":("dummy_ment_file", "dummy_ent_file", "dummy_ent_tokens_file")}

		self.train_domains = ["dummy"],
		self.dev_domains = ["dummy"],
		self.mention_file_template = "",
		self.entity_file_template = ""
		self.entity_token_file_template = "",


		self.mode = "train"
		self.debug_w_small_data = 0

		# Model/Optimization specific params
		self.use_GPU  		= True
		self.num_gpus		= 1
		self.strategy		= ""

		self.model_type	= "" # Choose between bi-encoder, cross-encoder
		self.cross_enc_type = "default"
		self.bi_enc_type = "separate" # Use "separate" encoder for query/input/mention and label/entity or "shared" encoder
		self.bert_model = "" # Choose type of bert model - bert-uncased-large etc
		self.bert_args = {} # Some arguments to pass to bert-model when initializing
		self.lowercase = True # Use lowercase BERT tokenizer
		self.shuffle_data = True # Shuffle data during training
		self.path_to_model = ""
		self.encoder_wrapper_config = ""
		self.joint_train_alpha = 1.0 # parameter to combine cross-enccoder and biencoder loss under parameter sharing settings
		self.mutual_distill_alpha = 0.0 # parameter to combine cross-enccoder and biencoder loss under parameter sharing settings w/ mutual distillation loss. 0 indicates no loss from mutual distillation
		self.use_all_layers = False # Compute loss using representation from all layers of BERT model

		self.num_epochs = 4
		self.warmup_proportion = 0.01
		self.train_batch_size = 16
		self.grad_acc_steps = 4
		self.max_grad_norm = 1.
		self.loss_type = "ce"
		self.hinge_margin = 0.5
		self.topk_ce = 10
		self.dist_to_prob_method = "negate_linear"
		self.reload_dataloaders_every_n_epochs = 0
		self.ckpt_metric = "loss"
		self.num_top_k_ckpts = 2
		self.dump_data = 0 # Dump data into a pkl file

		self.neg_strategy = "dummy" # Strategy for choosing negatives per input for a cross-/bi-encoder model
		self.num_negs = 63 # Number of negatives per input when using a cross-/bi-encoder model
		self.neg_mine_interval = 1000
		self.neg_mine_bienc_model_file = ""
		self.init_num_negs = 200
		self.num_neg_splits = 1 # Number of smaller batches to split orginal negatives into. Eg if value 2, and num_negs=64, then one training instacen with 64 negs is split into 2 instances with 32 negs each.
		self.use_top_negs = True # Use top-k negs from retriever. If False then sample from top negs from retriever. Use for passage retrieval datasets

		# Parameters for multi-label datasets
		self.total_labels_per_input = 64
		self.pos_strategy = "keep_pos_together"
		self.max_pos_labels = 32
		self.train_precomp_top_labels_fname = "" # File storing information about top-labels for each datapoint for dev data
		self.dev_precomp_top_labels_fname = "" # File storing information about top-labels for each datapoint for dev data

		# Parameter for NSW search based training for crossencoder models
		self.nsw_max_nbrs = 10
		self.nsw_num_paths = 4
		self.nsw_embed_type = ""
		self.num_negs_per_node = 8
		self.dist_cutoff = 3
		self.nsw_beamsize = 2
		self.nsw_comp_budget = 250
		self.n_anchor_ments = 100
		self.nsw_metric = "l2"

		# Parameters for distillation
		# self.distil_fname = {"dummy_domain": "dummy_file_with_labels_n_scores"} # Dict containing info about labels and their scores to use for distillation
		self.ent_w_score_file_template = "" # Template name for file w/ info about labels and their scores to use for distillation. Need to fill in domain name using string format option for this to work
		self.train_ent_w_score_file_template = "" # Useful when train and dev files have to be different name formats
		self.dev_ent_w_score_file_template = "" # Useful when train and dev files have to be different name format

		self.distill_n_labels = 64 # Number of labels per example to use in distillation training
		self.ent_distill_pair_method = "consec" # Param to control how we pair entities when only training entity model during distillation
		self.ent_distill_pair_sim = "neg_diff" # How to measure similarity b/w two entities using their score

		## BERT model specific params
		self.embed_dim = 768
		self.pooling_type = "" # Pooling on top of encoder layer to obtain input/label embedding
		self.add_linear_layer = False
		self.max_input_len = 128
		self.max_label_len = 128



		# Training specific params
		self.data_parallel = False

		self.type_optimization = ""
		self.learning_rate = 0.00001
		self.weight_decay = 0.01
		self.fp16 = False

		self.ckpt_path = ""
		# Eval specific
		self.eval_batch_size = 64

		if filename is not None:
			with open(filename) as fin:
				param_dict = json.load(fin)
			# for key,val in param_dict.items():
				# assert key in self.__dict__, f"Config file has param = {key} with val = {param_dict[key]} but this param is not defind in config class"

			self.__dict__.update({key:val for key,val in param_dict.items() if key in self.__dict__})
			extra_params = {key:val for key,val in param_dict.items() if key not in self.__dict__}
			if len(extra_params) > 0:
				warnings.warn(f"\n\nExtra params in config dict {extra_params}\n\n")
			# self.__dict__.update(param_dict)

		self.torch_seed 	= None
		self.np_seed 		= None
		self.cuda_seed 		= None
		self.update_random_seeds(self.seed)
		# self.device = torch.device("cuda" if self.cuda else "cpu")

	# @classmethod
	# def load_from_dict(cls, param_dict):
	# 	temp_config = cls()
	# 	temp_config.__dict__.update(param_dict)
	#
	# def update_from_dict(self, param_dict):
	# 	self.__dict__.update(param_dict)

	@property
	def cuda(self):
		return self.use_GPU and torch.cuda.is_available()

	@property
	def device(self):
		return torch.device("cuda" if self.cuda else "cpu")

	@property
	def result_dir(self):

		result_dir = "{base}/d={d}/{prefix}m={m}_l={l}_neg={neg}_s={s}{misc}".format(
			base=self.base_res_dir + "/" + self.exp_id if self.exp_id != ""
													   else self.base_res_dir,
			prefix=self.res_dir_prefix,
			d=self.data_type,
			m=self.model_type,
			l=self.loss_type,
			neg=self.neg_strategy,
			s=self.seed,
			misc="_{}".format(self.misc) if self.misc != "" else "")

		return result_dir

	@property
	def model_dir(self):
		return os.path.join(self.result_dir, "model")

	def update_random_seeds(self, random_seed):

		self.seed = random_seed
		random.seed(random_seed)

		self.torch_seed  = random.randint(0, 1000)
		self.np_seed     = random.randint(0, 1000)
		self.cuda_seed   = random.randint(0, 1000)

		torch.manual_seed(self.torch_seed)
		np.random.seed(self.np_seed)
		if self.use_GPU and torch.cuda.is_available():
			torch.cuda.manual_seed(self.cuda_seed)


class GradientBasedInfConfig(BaseConfig):
	def __init__(self, filename=None):
		
		super(GradientBasedInfConfig, self).__init__(filename=filename)
		self.config_name 	= filename
		
		self.save_code		= True
		self.base_res_dir	= "../../results"
		self.exp_id			= ""
		self.res_dir_prefix	= "" # Prefix to add to result dir name
		self.misc			= ""
		self.data_dir		= "../../data/zeshel"
		
		self.seed 			= 1234
		self.use_GPU 		= True
		
		# Model specific params
		self.bi_model_file = "" # Checkpoint file for a bi-encoder model
		self.cross_model_file = "" # Checkpoint file for a cross-encoder model
		self.embed_dim	= 768 # Embedding dim token embeddings

		
		self.data_name 	= ""
		self.n_ment		= 1
		
		self.param_type = "" # How to parameterize the model for gradient-based inference
		self.optimizer_type = "AdamW"
		self.num_search_steps = 10
		self.lr			= 0.1
		self.quant_method  = "" # Method for discretizing a soft label embedding to a discrete label
		self.reinit_w_quant_interval = 0 # Quantize soft label after every reinint_w_quant_interval search steps and re-initialize search parameters based on this label
		self.entropy_reg_alpha = 0.0 # Weight given to entropy-based regularization term (to encourage low-entropy solutions)
		
		self.init_method = "" # Method for choosing initial entity for starting search
		self.smooth_alpha  = 0.00000001 # Smoothening parameter when initializing using a discrete label embedding
		
		self.update_params_using_file(filename=filename)
		
		self.torch_seed 	= None
		self.np_seed 		= None
		self.cuda_seed 		= None
		self.update_random_seeds(self.seed)
		
		
	def update_params_using_file(self, filename):
		if filename is not None:
			with open(filename) as fin:
				param_dict = json.load(fin)
				
			self.__dict__.update({key:val for key,val in param_dict.items() if key in self.__dict__})
			extra_params = {key:val for key,val in param_dict.items() if key not in self.__dict__}
			if len(extra_params) > 0:
				warnings.warn(f"\n\nExtra params in config dict {extra_params}\n\n")
		
	@property
	def cuda(self):
		return self.use_GPU and torch.cuda.is_available()
	
	@property
	def device(self):
		return torch.device("cuda" if self.cuda else "cpu")
	
	@property
	def result_dir(self):

		result_dir = "{base}/gbi_d={d}/{prefix}nm={n_ment}_param={param_type}_q={quant}_init={init}_lr={lr}_smalpha={smooth_alpha}_ns={num_search_step}_s={s}{misc}".format(
			base=self.base_res_dir + "/" + self.exp_id if self.exp_id != ""
													   else self.base_res_dir,
			prefix=self.res_dir_prefix,
			param_type=self.param_type,
			d=self.data_name,
			n_ment=self.n_ment,
			quant=self.quant_method,
			lr=self.lr,
			num_search_step=self.num_search_steps,
			init=self.init_method,
			smooth_alpha=self.smooth_alpha,
			s=self.seed,
			misc="_{}".format(self.misc) if self.misc != "" else "")
		
		return result_dir
		
	def update_random_seeds(self, random_seed):
	
		self.seed = random_seed
		random.seed(random_seed)
		
		self.torch_seed  = random.randint(0, 1000)
		self.np_seed     = random.randint(0, 1000)
		self.cuda_seed   = random.randint(0, 1000)
		
		torch.manual_seed(self.torch_seed)
		np.random.seed(self.np_seed)
		if self.use_GPU and torch.cuda.is_available():
			torch.cuda.manual_seed(self.cuda_seed)

	def validate_params(self):
		"""
		Check that all parameters are taking valid values and are compatible with each other
		:return:
		"""
		
		assert self.quant_method in ["concat", "viterbi", "unigram_greedy", "label_greedy"]
		assert self.bi_model_file == ""
		assert self.param_type in ["free_embeds", "per_pos_weights", "per_label_weight"]
		
		if self.param_type == "per_label_weight":
			assert self.quant_method in ["label_greedy"]

		if self.param_type == "free_embeds":
			assert self.quant_method in ["concat", "unigram_greedy", "viterbi"]
			
		if self.param_type == "per_pos_weights":
			assert self.quant_method in ["unigram_greedy", "viterbi", "label_greedy"]
		
		if self.entropy_reg_alpha != 0.0:
			assert self.param_type in ["per_pos_weights", "per_label_weight"]
			
	
def filter_json(the_dict):
	res = {}
	for k in the_dict.keys():
		if type(the_dict[k]) is str or \
				type(the_dict[k]) is float or \
				type(the_dict[k]) is int or \
				type(the_dict[k]) is list or \
				type(the_dict[k]) is bool or \
				the_dict[k] is None:
			res[k] = the_dict[k]
		elif type(the_dict[k]) is dict:
			res[k] = filter_json(the_dict[k])
	return res
