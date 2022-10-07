import os
import gc
import csv
import sys
import math
import copy
import time
import json
import glob
import wandb
import torch
import pprint
import pickle
import logging
import warnings
import numpy as np


from pathlib import Path
from collections import defaultdict
from IPython import embed
# from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist

from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.config import Config
from utils.data_process import load_raw_data, get_dataloader, XMCDataset


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def load_pairwise_model(config):
	
	if config.model_type == "bi_enc":
		pairwise_model = BiEncoderWrapper(config)
		return pairwise_model
	elif config.model_type == "cross_enc":
		pairwise_model = CrossEncoderWrapper(config)
		return pairwise_model
	else:
		raise NotImplementedError(f"Support for model type = {config.model_type} not implemented")


class EntLinkData(LightningDataModule):
	
	def __init__(self, config):
		super(EntLinkData, self).__init__()
		self.config = config
		self.raw_dev_data = {}
		self.raw_train_data = {}
		
	# def prepare_data(self):
	# 	# Do not assign state in this function i.e. do not do self.x = y
	# 	pass
	
	def setup(self, stage=None):
		LOGGER.info("Inside setup function in DataModule")
		
		self.raw_train_data	= load_raw_data(config=self.config, data_split_type="train")
		self.raw_dev_data	= load_raw_data(config=self.config, data_split_type="dev")
		
		if self.config.debug_w_small_data:
			self.raw_train_data = {domain:(ments[:100], ents) for domain, (ments, ents) in self.raw_train_data.items()}
			self.raw_dev_data 	= {domain:(ments[:100], ents) for domain, (ments, ents) in self.raw_dev_data.items()}
		
		LOGGER.info("Finished setup function in DataModule")
	
	def val_dataloader(self):
		# reset_neg_strategy = False
		# if self.config.neg_strategy == "precomp" and self.trainer.current_epoch == 0:
		# 	LOGGER.info(f"\n\nMining negs using biencoder for first epoch as neg_strategy=precomp\n\n")
		# 	self.config.neg_strategy = "bienc_hard_negs"
		# 	reset_neg_strategy = True
		
		# # Hack to use distilled biencoder from first epoch onward
		# if self.trainer.current_epoch == 1:
		# 	LOGGER.info("Updating neg_mine_bienc_model_file to a distilled biencoder file")
		# 	self.config.neg_mine_bienc_model_file = "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_all_data/model/model-3-12318.0-1.92.ckpt"
		
		LOGGER.info("\t\tAbout to start loading biencoder and crossencoder models as needed")
		biencoder = self.get_bienc_model()
		crossencoder = self.get_cross_enc_model()
		LOGGER.info("\t\tFinished biencoder and crossencoder models as needed")
		
		# reranker_batch_size -> computing by adjusting eval_batch_size based on init_num_negs and num_negs param
		reranker_batch_size = int((self.config.num_negs+1)*self.config.train_batch_size/self.config.init_num_negs)
		reranker_batch_size = max(1, reranker_batch_size)
		
		bienc_in_train_mode = biencoder.training if biencoder else False
		crossenc_in_train_mode = crossencoder.training if crossencoder else False
		
		if biencoder: biencoder.eval()
		if crossencoder: crossencoder.eval()
		
		LOGGER.info(f"\n\n\t\tLoading validation data in DataModule w/ reranker_batch_size = {reranker_batch_size}\n\n")
		dump_dir = f"{self.config.result_dir}/data_dump/epoch_{self.trainer.current_epoch}" if self.config.dump_data else None
		dev_dataloader = get_dataloader(
			split_type="dev",
			raw_data=self.raw_dev_data,
			config=self.config,
			batch_size=self.config.eval_batch_size,
			shuffle_data=False,
			biencoder=biencoder,
			reranker=crossencoder,
			reranker_batch_size=reranker_batch_size,
			dump_dir=dump_dir
		)
		if biencoder and bienc_in_train_mode: biencoder.train()
		if crossencoder and crossenc_in_train_mode: crossencoder.train()
		
		
		torch.cuda.empty_cache()
		LOGGER.info("Finished loading validation data")
		return dev_dataloader
	
	def train_dataloader(self):
		# reset_neg_strategy = False
		# if self.config.neg_strategy == "precomp" and self.trainer.current_epoch == 0:
		# 	LOGGER.info(f"\n\nMining negs using biencoder for first epoch as neg_strategy=precomp\n\n")
		# 	self.config.neg_strategy = "bienc_hard_negs"
		# 	reset_neg_strategy = True
		
		# # Hack to use distilled biencoder from first epoch onward
		# if self.trainer.current_epoch == 1:
		# 	LOGGER.info("Updating neg_mine_bienc_model_file to a distilled biencoder file")
		# 	self.config.neg_mine_bienc_model_file = "../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_all_data/model/model-3-12318.0-1.92.ckpt"
		
		LOGGER.info("\t\tAbout to start loading biencoder and crossencoder models as needed")
		biencoder = self.get_bienc_model()
		crossencoder = self.get_cross_enc_model()
		LOGGER.info("\t\tFinished biencoder and crossencoder models as needed")
		LOGGER.info("\t\tAll GPUs synced")
		
		# Load data in pytorch data loader.
		# Since we will take a gradient step after self.config.grad_acc_steps,
		# effective_batch_size given to data_loader = int(config.train_batch_size/self.config.grad_acc_steps)
		
		# reranker_batch_size -> computing by adjusting eval_batch_size based on init_num_negs and num_negs param
		reranker_batch_size = int((self.config.num_negs+1)*self.config.train_batch_size/self.config.init_num_negs)
		reranker_batch_size = max(1, reranker_batch_size)
		
		bienc_in_train_mode = biencoder.training if biencoder else False
		crossenc_in_train_mode = crossencoder.training if crossencoder else False
		if biencoder: biencoder.eval()
		if crossencoder: crossencoder.eval()
		
		
		dump_dir = f"{self.config.result_dir}/data_dump/epoch_{self.trainer.current_epoch}" if self.config.dump_data else None
		LOGGER.info(f"\n\n\t\tLoading training data in DataModule w/ reranker_batch_size = {reranker_batch_size}\n\n")
		train_dataloader = get_dataloader(
			split_type="train",
			config=self.config,
			raw_data=self.raw_train_data,
			batch_size=int(self.config.train_batch_size/self.config.grad_acc_steps),
			shuffle_data=self.config.shuffle_data,
			biencoder=biencoder,
			reranker=crossencoder,
			reranker_batch_size=reranker_batch_size,
			dump_dir=dump_dir
		)
		if biencoder and bienc_in_train_mode: biencoder.train()
		if crossencoder and crossenc_in_train_mode: crossencoder.train()
		
		torch.cuda.empty_cache()
		LOGGER.info("Finished loading training data")
		return train_dataloader
	
	@property
	def train_data_len(self):
		"""
		Returns number of batches in training data.
		This assumes that a batch of size b will contain positive and negative entities for b unique mentions,
		and that each mention contributes exactly one training datapoint to train_dataloader
		:return:
		"""
		if self.config.data_type in ["ent_link", "ent_link_ce"]:
			batch_size = int(self.config.train_batch_size/self.config.grad_acc_steps)
			total_ments = np.sum([len(ment_data) for domain, (ment_data, ent_data) in self.raw_train_data.items()])
			num_batches = int(total_ments/batch_size)
			return num_batches
		elif self.config.data_type == "nq":
			batch_size = int(self.config.train_batch_size/self.config.grad_acc_steps)
			total_ments = np.sum([len(data) for domain, data in self.raw_train_data.items()])
			num_batches = int(total_ments/batch_size)
			return num_batches
		elif self.config.data_type in ["xmc"]:
			batch_size = int(self.config.train_batch_size/self.config.grad_acc_steps)
			assert isinstance(self.raw_train_data, XMCDataset)
			total_inputs = len(self.raw_train_data.data)
			num_batches = int(total_inputs/batch_size)
			return num_batches
		else:
			raise NotImplementedError(f"Data type = {self.config.data_type} not supported")
	
	def get_bienc_model(self):
		if self.config.model_type == "cross_enc":
			load_bienc_model = self.config.neg_strategy in ["bienc_distill", "bienc_hard_negs", "bienc_hard_negs_w_rerank", "bienc_nsw_search", "bienc_hard_negs_w_knn_rank"] \
							or (self.config.nsw_embed_type == "bienc" and self.config.neg_strategy in ["nsw_graph_path", "nsw_graph_rank", "hard_negs_w_rank"])
		
			if load_bienc_model and os.path.isfile(self.config.neg_mine_bienc_model_file):
				LOGGER.info(f"Loading biencoder model from {self.config.neg_mine_bienc_model_file}")
				neg_mine_bienc_model = load_pairwise_model(Config(self.config.neg_mine_bienc_model_file)) \
					if self.config.neg_mine_bienc_model_file.endswith("json") \
					else BiEncoderWrapper.load_from_checkpoint(self.config.neg_mine_bienc_model_file)
				
				LOGGER.info(f"Finished loading biencoder model from {self.config.neg_mine_bienc_model_file}")
				return neg_mine_bienc_model
			elif load_bienc_model and (self.config.joint_train_alpha < 1) and (self.trainer.current_epoch > 0):
				LOGGER.info(f"Returning cross-encoder model that can also be used as a biencoder as "
							f"self.config.joint_train_alpha = {self.config.joint_train_alpha} < 1 and current_epoch={self.trainer.current_epoch}")
				return self.trainer.model
			else:
				LOGGER.info(f"Biencoder model will not be loaded as load_bienc_model = {load_bienc_model} and "
							f"neg_mine_bienc_model exists = {os.path.isfile(self.config.neg_mine_bienc_model_file)} "
							f"self.config.joint_train_alpha = {self.config.joint_train_alpha} and current_epoch={self.trainer.current_epoch}")
				return None
				
		elif self.config.model_type == "bi_enc":
			# Use model only if epoch > 0 or pretrained model was specified in self.config.path_to_model
			if self.trainer.current_epoch > 0 or os.path.isfile(self.config.path_to_model):
				LOGGER.info(f"Returning current biencoder model")
				return self.trainer.model
			else:
				LOGGER.info(f"Returning None for biencoder model as current_epoch>0 ->{self.trainer.current_epoch > 0},"
							f" path_to_model file exists -> {os.path.isfile(self.config.path_to_model)}")
				return None
		else:
			raise Exception(f"Model type = {self.config.model_type} {type(self.trainer.model)} not supported")
		
	def get_cross_enc_model(self):
		"""
		Loads and returns a crossencoder model
			if neg_strategy requires a crossencoder model and (either epoch >= 1 or a pretrained model was specified using self.config.path_to_model)
			else return None
		:return:
		"""
		load_cross_enc_model = (self.config.neg_strategy in ["bienc_hard_negs_w_rerank", "tfidf_hard_negs_w_rerank", "bienc_nsw_search", "tfidf_nsw_search"]) \
							   and (self.trainer.current_epoch > 0 or os.path.isfile(self.config.path_to_model))
		
		if load_cross_enc_model and self.config.model_type  == "cross_enc":
			LOGGER.info(f"Returning current crossencoder model")
			return self.trainer.model
		else:
			LOGGER.info(f"Returning None for crossencoder model as neg_strategy= {self.config.neg_strategy},"
						f"trainer.epoch = {self.trainer.current_epoch}, "
						f"path_to_model exists = {os.path.isfile(self.config.path_to_model)}, "
						f"is instance of cross_encoder = {self.config.model_type}")
			return None
			
		


class BasePairwiseTrainer(object):
	"""
	Trainer class to train a pairwise similarity model
	"""
	def __init__(self, config):
		
		assert isinstance(config, Config)
		self.config = config
		
		LOGGER.addHandler(logging.FileHandler(f"{self.config.result_dir}/log_file.txt"))
		
		# wandb initialization
		config_dict = self.config.__dict__
		config_dict["CUDA_DEVICE"] = os.environ["CUDA_VISIBLE_DEVICES"]
		
		os.environ["WANDB_API_KEY"] = "6ae7d53ecce3f7d824317087c3973ebd50e29bb3"
		self.wandb_logger = WandbLogger(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict)
		try:
			wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict)
		except Exception as e:
			try:
				LOGGER.info(f"Trying with wandb.Settings(start_method=fork) as error = {e} as raised")
				wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict,
						   settings=wandb.Settings(start_method="fork"))
			except Exception as e:
				LOGGER.info(f"Error raised = {e}")
				LOGGER.info("Running wandb in offline mode")
				wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict,
						   mode="offline")
		
		
		# Create datamodule
		self.data	  	= EntLinkData(config=config)
		
		# Load the model and optimizer
		self.pw_model 	= load_pairwise_model(config=self.config)
		
		# wandb.watch(self.pw_model, log_freq=self.config.print_interval*self.config.grad_acc_steps)
		
		
	def __str__(self):
		print_str = pprint.pformat(self.config.to_json())
		print_str += f"Model parameters:{self.pw_model}"
		return print_str
		
		
	def train(self):
		seed_everything(self.config.seed, workers=True)

		if self.config.neg_strategy in ["precomp", "random"]:
			pass
		elif self.config.model_type == "bi_enc" and self.config.neg_strategy in ["bienc_hard_negs", "top_ce_as_pos_w_bienc_hard_negs", "top_ce_w_bienc_hard_negs_trp", "top_ce_w_bienc_hard_negs_ml"]:
			assert self.config.reload_dataloaders_every_n_epochs == 1, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}"
		elif self.config.model_type == "cross_enc" and self.config.joint_train_alpha == 1 and self.config.neg_strategy in ["bienc_hard_negs_w_rerank", "tfidf_hard_negs_w_rerank", "bienc_nsw_search", "tfidf_nsw_search"]:
			assert self.config.reload_dataloaders_every_n_epochs == 1, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}, joint_train_alpha = {self.config.joint_train_alpha}"
		elif self.config.model_type == "cross_enc" and self.config.joint_train_alpha < 1 and self.config.neg_strategy in ["bienc_hard_negs", "top_ce_as_pos_w_bienc_hard_negs"]:
			assert self.config.reload_dataloaders_every_n_epochs == 1, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}, joint_train_alpha = {self.config.joint_train_alpha}"
		else:
			assert self.config.reload_dataloaders_every_n_epochs == 0, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}, joint_train_alpha = {self.config.joint_train_alpha}"
		
		metric_to_monitor =  f"dev_{self.config.ckpt_metric}"
		checkpoint_callback = ModelCheckpoint(
			save_top_k=self.config.num_top_k_ckpts,
			monitor=metric_to_monitor,
			mode="min",
			auto_insert_metric_name=False,
			save_last=False, # Determines if we save another copy of checkpoint
			save_weights_only=False,
			dirpath=self.config.model_dir,
			filename="model-{epoch}-{train_step}-{" + metric_to_monitor + ":.2f}",
			save_on_train_epoch_end=False # Set to False to run check-pointing checks at the end of the val loop
		)
		checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
		
		
		end_of_epoch_checkpoint_callback = ModelCheckpoint(
			auto_insert_metric_name=False,
			save_last=False,
			save_weights_only=False,
			dirpath=self.config.model_dir,
			filename="eoe-{epoch}-last",
			save_on_train_epoch_end=True # Set to True to run check-pointing checks at end of each epoch
		)
		end_of_epoch_checkpoint_callback.CHECKPOINT_NAME_LAST = "eoe-{epoch}-last"
		
		# early_stop_callback = EarlyStopping(monitor="dev_loss", mode="min")
		lr_monitor = LearningRateMonitor(logging_interval='step')
		strategy = self.config.strategy if self.config.strategy in ["dp", "ddp", "ddp_spawn"] else None
		assert self.config.num_gpus <= 1 or strategy is not None, f"Can not pass more than 1 GPU with strategy = {strategy}"
		
		# auto_lr_find = self.config.reload_dataloaders_every_n_epochs == 0
		auto_lr_find = False
		
		val_check_interval = int(self.config.eval_interval*self.config.grad_acc_steps) if self.config.eval_interval > 1 else self.config.eval_interval
		trainer = Trainer(
			gpus=self.config.num_gpus,
			strategy=strategy,
			default_root_dir=self.config.result_dir,
			max_epochs=self.config.num_epochs,
			max_time=self.config.max_time,
			accumulate_grad_batches=self.config.grad_acc_steps,
			reload_dataloaders_every_n_epochs=self.config.reload_dataloaders_every_n_epochs,
			fast_dev_run=self.config.fast_dev_run,
			val_check_interval=val_check_interval,
			gradient_clip_val=self.config.max_grad_norm,
			callbacks=[lr_monitor, checkpoint_callback, end_of_epoch_checkpoint_callback],
			logger=self.wandb_logger,
			auto_lr_find=auto_lr_find,
			profiler="simple",
			num_sanity_val_steps=0
		)
		#
		# set auto_lr_find to true to enable this -> trainer.tune(model=self.pw_model, datamodule=self.data) # TODO: Check model loading behaviour with tune function
		if auto_lr_find:
			trainer.tune(model=self.pw_model, datamodule=self.data)
			LOGGER.info("\n\n\t\tFinished tuning learning rate\n\n")
			
		ckpt_path = self.config.ckpt_path if self.config.ckpt_path != "" else None
		trainer.fit(model=self.pw_model, datamodule=self.data, ckpt_path=ckpt_path)
