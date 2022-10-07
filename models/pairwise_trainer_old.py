# import os
# import csv
# import sys
# import math
# import copy
# import time
# import json
# import wandb
# import torch
# import pprint
# import logging
# import numpy as np
#
# from tqdm import tqdm
# from IPython import embed
# from pathlib import Path
# # from accelerate import Accelerator
# from collections import defaultdict
#
#
# from models.biencoder import BiEncoderWrapper
# from models.crossencoder import CrossEncoderWrapper
# from utils.config import Config
# from utils.data_process import load_raw_data, get_dataloader, NSWDataset
# from utils.optimizer import get_bert_optimizer, get_scheduler
#
#
# logging.basicConfig(
# 	stream=sys.stderr,
# 	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
# 	datefmt="%d/%m/%Y %H:%M:%S",
# 	level=logging.INFO,
# )
#
# LOGGER = logging.getLogger(__name__)
#
#
# def load_pairwise_model(config):
#
# 	if config.model_type == "bi_enc":
# 		pairwise_model = BiEncoderWrapper(config)
# 		return pairwise_model
# 	elif config.model_type == "cross_enc":
# 		pairwise_model = CrossEncoderWrapper(config)
# 		return pairwise_model
# 	else:
# 		raise NotImplementedError(f"Support for model type = {config.model_type} not implemented")
#
#
# class BasePairwiseTrainer(object):
# 	"""
# 	Trainer class to train a pairwise similarity model
# 	"""
# 	def __init__(self, config):
#
# 		assert isinstance(config, Config)
# 		self.config = config
#
# 		# accelerator = Accelerator()
# 		# self.config.device = accelerator.device
#
# 		LOGGER.addHandler(logging.FileHandler(f"{self.config.result_dir}/log_file.txt"))
#
# 		# wandb initialization
# 		config_dict = self.config.__dict__
# 		config_dict["CUDA_DEVICE"] = os.environ["CUDA_VISIBLE_DEVICES"]
# 		wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict)
#
# 		# TODO:
# 		# 1) When should we fix random seeds? In config or inside trainer?
#
# 		if self.config.neg_mine_bienc_model_file != "" and os.path.isfile(self.config.neg_mine_bienc_model_file):
# 			neg_mine_bienc_model = load_pairwise_model(Config(self.config.neg_mine_bienc_model_file))
# 		else:
# 			neg_mine_bienc_model = None
#
#
# 		# Load training data.
# 		self.raw_train_data	= load_raw_data(config=self.config, data_split_type="train")
#
# 		# Load data in pytorch data loader.
# 		# Since we will take a gradient step after self.config.grad_acc_steps,
# 		# effective_batch_size = int(config.train_batch_size/self.config.grad_acc_steps)
# 		self.train_dataloader = get_dataloader(raw_data=self.raw_train_data, config=self.config,
# 											   batch_size=int(self.config.train_batch_size/self.config.grad_acc_steps),
# 											   biencoder=neg_mine_bienc_model)
#
# 		# Load validation data.
# 		self.raw_dev_data	= load_raw_data(config=self.config, data_split_type="dev")
# 		self.dev_dataloader = get_dataloader(raw_data=self.raw_dev_data, config=self.config,
# 											 batch_size=self.config.eval_batch_size,
# 											 biencoder=neg_mine_bienc_model)
#
# 		# Load the model and optimizer
# 		self.pw_model  = load_pairwise_model(config=self.config)
#
# 		self.optimizer = None
# 		self.scheduler = None
# 		self.reset_optimizer()
#
# 		wandb.watch(self.pw_model, log_freq=self.config.print_interval*self.config.grad_acc_steps)
# 		# self.pw_model, self.optimizer, self.train_dataloader = accelerator.prepare(self.pw_model, self.optimizer, self.train_dataloader)
# 		# self.dev_dataloader = accelerator.prepare(self.dev_dataloader)
#
#
# 	def __str__(self):
# 		print_str = pprint.pformat(self.config.to_json())
# 		print_str += f"Model parameters:{self.pw_model}"
# 		print_str += f"\nOptimizer :{self.optimizer}\n"
# 		# print_str += f"\nOptimizer Param Groups:{self.optimizer.param_groups}\n"
#
# 		return print_str
#
# 	def reset_optimizer(self):
#
# 		self.optimizer = get_bert_optimizer(models=[self.pw_model],
# 											type_optimization=self.config.type_optimization,
# 											learning_rate=self.config.learning_rate,
# 											weight_decay=self.config.weight_decay,
# 											optimizer_type="AdamW")
#
# 		self.scheduler = get_scheduler(optimizer=self.optimizer,
# 									   epochs=self.config.num_epochs,
# 									   warmup_proportion=self.config.warmup_proportion,
# 									   len_data=len(self.train_dataloader) if isinstance(self.train_dataloader, NSWDataset) else len(self.train_dataloader.dataset),
# 									   batch_size=self.config.train_batch_size,
# 									   grad_acc_steps=self.config.grad_acc_steps)
#
#
# 	def save_model(self, res_dir):
#
# 		# TODO: Add support for saving optimizer state and continue training using a saved model
#
# 		Path(res_dir).mkdir(exist_ok=True, parents=True)
# 		self.pw_model.save_model(res_dir=res_dir)
# 		self.config.save_config(res_dir=res_dir)
#
#
# 	def load_model(self):
# 		# TODO: Add support for loading optimizer state and continue training using a saved model
# 		return load_pairwise_model(config=self.config)
#
#
# 	def update_random_seeds(self):
# 		self.config.update_random_seeds(random_seed=self.config.seed)
#
# 	def train(self):
# 		raise NotImplementedError
#
#
# 	def debug(self):
# 		raise NotImplementedError
#
# 	def calc_loss(self, data, type_list):
# 		raise NotImplementedError
#
# 	def stop_training(self, epoch, all_losses):
# 		raise NotImplementedError
#
# 	@staticmethod
# 	def write_metrics(all_losses, all_scores, curr_result_dir, xlabel):
# 		pass
#
# 	def read_metrics(self, filename=None):
# 		pass
#
# 	@staticmethod
# 	def evaluate(pw_model, eval_dataloader, **kwargs):
# 		raise NotImplementedError
#
#
# class BiEncPairwiseTrainer(BasePairwiseTrainer):
# 	"""
# 	Trainer class to train a pairwise similarity model using bi-encoders
# 	"""
# 	def __init__(self, config):
# 		super(BiEncPairwiseTrainer, self).__init__(config=config)
#
#
# 	def train(self):
#
# 		try:
# 			for split_type in ["train", "dev"]:
# 				wandb.define_metric(f"{split_type}/step")
# 				wandb.define_metric(f"{split_type}/*", step_metric=f"{split_type}/step")
#
# 			self.update_random_seeds()
# 			self.pw_model.train()
# 			self.optimizer.zero_grad()
#
# 			avg_train_loss = 0.
# 			best_dev_configs = {}
# 			for epoch_iter in range(self.config.num_epochs):
#
# 				best_dev_loss = None # Setting best_dev_loss to None here so that we can find best model each epoch
#
# 				# Update dataloader using negatives from most wrt most recent model
# 				if self.config.neg_strategy == "bienc_hard_negs" and epoch_iter >= 1:
# 					LOGGER.info("Mining hard negatives wrt current model parameters")
# 					# assert self.config.shuffle_data, "Data shuffling should be turned on because we are terminating current " \
# 					# 								 "epochs prematurely. Without data shuffling we will always" \
# 					# 								 "iterate on a small subset of data"
#
# 					# Load data again with hard negatives
# 					self.train_dataloader = get_dataloader(raw_data=self.raw_train_data, config=self.config,
# 														   batch_size=int(self.config.train_batch_size/self.config.grad_acc_steps),
# 														   biencoder=self.pw_model)
#
# 					self.dev_dataloader = get_dataloader(raw_data=self.raw_dev_data, config=self.config,
# 														   batch_size=self.config.eval_batch_size,
# 														   biencoder=self.pw_model)
#
# 				for step, batch in enumerate(self.train_dataloader):
#
# 					if len(batch) == 2:
# 						batch_input, batch_pos_labels = batch
# 						batch_neg_labels = None
# 					else:
# 						batch_input, batch_pos_labels, batch_neg_labels = batch
#
# 					batch_input = batch_input.to(self.pw_model.device)
# 					batch_pos_labels = batch_pos_labels.to(self.pw_model.device)
# 					batch_neg_labels = batch_neg_labels.to(self.pw_model.device) if batch_neg_labels is not None else None
#
# 					loss = self.pw_model(batch_input, batch_pos_labels, batch_neg_labels)
# 					avg_train_loss += loss.item()
#
# 					# Divide loss by self.config.grad_acc_steps is we are accumulating gradients
# 					loss = loss / self.config.grad_acc_steps if self.config.grad_acc_steps > 1 else loss
#
# 					# Compute gradients
# 					loss.backward()
#
# 					# Perform optimization step after accumulating gradients for self.config.grad_acc_steps steps
# 					if (step + 1) % self.config.grad_acc_steps == 0:
# 						torch.nn.utils.clip_grad_norm_(self.pw_model.parameters(), self.config.max_grad_norm)
# 						self.optimizer.step()
# 						self.scheduler.step()
# 						self.optimizer.zero_grad()
#
# 					# Eval on eval dataset
# 					if (step + 1) % (self.config.eval_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info("Evaluation on the development dataset")
# 						self.pw_model.eval()
# 						eval_res = self.evaluate(pw_model=self.pw_model, eval_dataloader=self.dev_dataloader)
# 						wandb.log({"dev/epoch": epoch_iter,
# 								   "dev/step": epoch_iter*len(self.train_dataloader) + step,
# 								   "dev/loss": eval_res["loss"]})
# 						self.pw_model.train()
#
# 						if best_dev_loss is None or eval_res["loss"] < best_dev_loss:
# 							LOGGER.info("***** Saving best model on dev set *****")
# 							best_dev_loss = eval_res["loss"]
# 							wandb.run.summary["best_dev_loss"] = best_dev_loss
# 							self.save_model(res_dir=os.path.join(self.config.model_dir, f"best_wrt_dev_{epoch_iter}"))
# 							best_dev_configs[epoch_iter] = copy.deepcopy(self.config)
#
# 					# print loss metrics
# 					if (step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info(
# 							f"Step {step} - epoch {epoch_iter}"
# 							f" average loss: {avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)}\n"
# 						)
# 						avg_train_loss = avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)
# 						wandb.log({"train/epoch": epoch_iter,
# 								   "train/step": epoch_iter*len(self.train_dataloader) + step,
# 								   "train/step_frac": float(step)/len(self.train_dataloader),
# 								   "train/loss": avg_train_loss})
# 						avg_train_loss = 0.
#
#
# 				LOGGER.info("***** Saving fine - tuned model *****")
# 				self.save_model(res_dir=os.path.join(self.config.model_dir, "curr_epoch"))
#
# 			# Pick best dev model amongst best models from each epoch
# 			final_config = copy.deepcopy(self.config)
# 			best_dev_model_loss = None
# 			best_dev_model_config = None
# 			for epoch_iter, curr_config in best_dev_configs.items():
# 				self.config = curr_config
# 				self.pw_model = self.load_model()
# 				self.pw_model.eval()
# 				curr_eval_res = self.evaluate(pw_model=self.pw_model, eval_dataloader=self.dev_dataloader)
# 				if best_dev_model_loss is None or curr_eval_res["loss"] < best_dev_model_loss:
# 					best_dev_model_loss = curr_eval_res["loss"]
# 					best_dev_model_config = curr_config
#
# 			self.save_model(res_dir=os.path.join(best_dev_model_config.model_dir, f"best_wrt_dev"))
#
# 			self.config =  final_config
# 		except Exception as e:
# 			embed()
# 			raise e
#
#
#
# 	def debug(self):
#
# 		input_text = [self.raw_train_data.data[0]["input"]]
# 		label_text = [self.raw_train_data.labels[0]]
#
# 		LOGGER.info(f"Input: {input_text}")
# 		LOGGER.info(f"Label: {label_text}")
#
# 		ans = self.pw_model.forward_w_raw(input_text=[], label_text=[])
#
# 		LOGGER.info(f"Ans: {ans}")
#
# 		embed()
#
# 	@staticmethod
# 	def evaluate(pw_model, eval_dataloader, **kwargs):
# 		"""
# 		Compute loss with given pairwise model and data
# 		:param pw_model:
# 		:param eval_dataloader:
# 		:return:
# 		"""
# 		pw_model.eval()
# 		eval_loss_list = []
#
# 		with torch.no_grad():
# 			for step, batch in enumerate(eval_dataloader):
#
# 				if len(batch) == 2:
# 					batch_input, batch_pos_labels = batch
# 					batch_neg_labels = None
# 				else:
# 					batch_input, batch_pos_labels, batch_neg_labels = batch
#
# 				batch_input = batch_input.to(pw_model.device)
# 				batch_pos_labels = batch_pos_labels.to(pw_model.device)
# 				batch_neg_labels = batch_neg_labels.to(pw_model.device) if batch_neg_labels is not None else None
#
# 				loss = pw_model(batch_input, batch_pos_labels, batch_neg_labels)
# 				eval_loss_list += [loss.item()]
#
# 		eval_loss = np.mean(eval_loss_list)
# 		return {"loss": eval_loss}
#
#
#
# class CrossEncPairwiseTrainer(BasePairwiseTrainer):
# 	"""
# 	Trainer class to train a pairwise similarity model using cross-encoders
# 	"""
# 	def __init__(self, config):
#
# 		super(CrossEncPairwiseTrainer, self).__init__(config=config)
#
# 	def train(self):
#
# 		try:
# 			for split_type in ["train", "dev"]:
# 				wandb.define_metric(f"{split_type}/step")
# 				wandb.define_metric(f"{split_type}/*", step_metric=f"{split_type}/step")
#
# 			self.update_random_seeds()
# 			self.pw_model.train()
# 			self.optimizer.zero_grad()
#
# 			avg_train_loss = 0.
# 			best_dev_loss = None
# 			for epoch_iter in range(self.config.num_epochs):
#
# 				for step, batch in enumerate(self.train_dataloader):
#
# 					batch_pos_pairs, batch_neg_pairs = batch
#
# 					batch_pos_pairs = batch_pos_pairs.to(self.pw_model.device)
# 					batch_neg_pairs = batch_neg_pairs.to(self.pw_model.device)
#
# 					# TODO: Implement more loss functions for cross-encoder - Like maybe pack pos and negative labels together per input?
# 					loss = self.pw_model(batch_pos_pairs, batch_neg_pairs, first_segment_end=self.config.max_input_len)
# 					avg_train_loss += loss.item()
#
# 					# Divide loss by self.config.grad_acc_steps is we are accumulating gradients
# 					loss = loss / self.config.grad_acc_steps if self.config.grad_acc_steps > 1 else loss
#
# 					# Compute gradients
# 					loss.backward()
#
# 					# Perform optimization step after accumulating gradients for self.config.grad_acc_steps steps
# 					if (step + 1) % self.config.grad_acc_steps == 0:
# 						torch.nn.utils.clip_grad_norm_(self.pw_model.parameters(), self.config.max_grad_norm)
# 						self.optimizer.step()
# 						self.scheduler.step()
# 						self.optimizer.zero_grad()
#
# 					# Eval on eval dataset
# 					if (step + 1) % (self.config.eval_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info("Evaluation on the development dataset")
# 						self.pw_model.eval()
# 						eval_res = self.evaluate(pw_model=self.pw_model, eval_dataloader=self.dev_dataloader,
# 												 first_segment_end=self.config.max_input_len)
# 						wandb.log({"dev/epoch": epoch_iter,
# 								   "dev/step": epoch_iter*len(self.train_dataloader) + step,
# 								   "dev/loss": eval_res["loss"]})
# 						self.pw_model.train()
#
# 						if best_dev_loss is None or eval_res["loss"] < best_dev_loss:
# 							LOGGER.info("***** Saving best model on dev set *****")
# 							best_dev_loss = eval_res["loss"]
# 							wandb.run.summary["best_dev_loss"] = best_dev_loss
# 							self.save_model(res_dir=os.path.join(self.config.model_dir, "best_wrt_dev"))
#
# 					# print loss metrics
# 					if (step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info(
# 							f"Step {step} - epoch {epoch_iter}"
# 							f" average loss: {avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)}\n"
# 						)
# 						avg_train_loss = avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)
# 						wandb.log({"train/epoch": epoch_iter,
# 								   "train/step": epoch_iter*len(self.train_dataloader) + step,
# 								   "train/step_frac": float(step)/len(self.train_dataloader),
# 								   "train/loss": avg_train_loss})
# 						avg_train_loss = 0.
#
#
# 				LOGGER.info("***** Saving fine - tuned model *****")
# 				self.save_model(res_dir=os.path.join(self.config.model_dir, "curr_epoch"))
#
# 		except Exception as e:
# 			embed()
# 			raise e
#
#
# 	@staticmethod
# 	def evaluate(pw_model, eval_dataloader, **kwargs):
# 		"""
# 		Compute loss with given pairwise model and data
# 		:param pw_model:
# 		:param eval_dataloader:
# 		:return:
# 		"""
# 		pw_model.eval()
# 		eval_loss_list = []
#
# 		for step, batch in enumerate(eval_dataloader):
#
# 			batch_pos_pairs, batch_neg_pairs = batch
#
# 			batch_pos_pairs = batch_pos_pairs.to(pw_model.device)
# 			batch_neg_pairs = batch_neg_pairs.to(pw_model.device)
#
# 			with torch.no_grad():
# 				loss = pw_model(batch_pos_pairs, batch_neg_pairs, first_segment_end=kwargs["first_segment_end"])
# 				eval_loss_list += [loss.item()]
#
# 		eval_loss = np.mean(eval_loss_list)
# 		return {"loss": eval_loss}
#
#
# class CrossEncPairwiseTrainerwNSW(BasePairwiseTrainer):
# 	"""
# 	Trainer class to train a pairwise similarity model using cross-encoders on top of NSW graps
# 	"""
# 	def __init__(self, config):
#
# 		super(CrossEncPairwiseTrainerwNSW, self).__init__(config=config)
#
# 	def train(self):
#
# 		try:
# 			for split_type in ["train", "dev"]:
# 				wandb.define_metric(f"{split_type}/step")
# 				wandb.define_metric(f"{split_type}/*", step_metric=f"{split_type}/step")
#
# 			self.update_random_seeds()
# 			self.pw_model.train()
# 			self.optimizer.zero_grad()
#
# 			# def _yield_path_level_pairs(dataloader):
# 			# 	from utils.data_process import NSWDataset
# 			# 	assert isinstance(dataloader, NSWDataset)
# 			# 	all_pos_neg_pairs = dataloader.all_pos_neg_pairs
# 			# 	all_pos_neg_pair_token_idxs = dataloader.all_pos_neg_pair_token_idxs
# 			# 	for curr_pos_neg_pairs, curr_pos_neg_pair_token_idxs in zip(all_pos_neg_pairs, all_pos_neg_pair_token_idxs):
# 			# 		for path_pos_neg_pairs, path_pos_neg_pair_token_idxs in zip(curr_pos_neg_pairs, curr_pos_neg_pair_token_idxs):
# 			# 			yield (path_pos_neg_pairs, path_pos_neg_pair_token_idxs)
#
# 			avg_train_loss = 0.
# 			best_dev_loss = None
# 			for epoch_iter in range(self.config.num_epochs):
# 				total_paths = len(self.train_dataloader)
# 				# for step, (path_pos_neg_pairs, path_pos_neg_pair_token_idxs) in enumerate(_yield_path_level_pairs(self.train_dataloader)):
# 				# 	path_pos_pair_token_idxs, path_neg_pair_token_idxs = path_pos_neg_pair_token_idxs
# 				# 	loss = self.pw_model.forward_w_nsw_wo_batch(path_pos_pair_idxs=path_pos_pair_token_idxs,
# 				# 												path_neg_pair_idxs=path_neg_pair_token_idxs,
# 				# 												first_segment_end=self.config.max_input_len)
#
# 				for step, batch in enumerate(self.train_dataloader):
# 					batch_pos_pairs, batch_neg_pairs = batch
#
# 					batch_pos_pairs = batch_pos_pairs.to(self.pw_model.device)
# 					batch_neg_pairs = batch_neg_pairs.to(self.pw_model.device)
#
# 					# TODO: Implement more loss functions for cross-encoder - Like maybe pack pos and negative labels together per input?
# 					loss = self.pw_model(batch_pos_pairs, batch_neg_pairs, first_segment_end=self.config.max_input_len)
#
# 					#
# 					# batch_pos_pairs, batch_neg_pairs, first_segment_end=self.config.max_input_len
# 					avg_train_loss += loss.item()
#
# 					# Divide loss by self.config.grad_acc_steps is we are accumulating gradients
# 					loss = loss / self.config.grad_acc_steps if self.config.grad_acc_steps > 1 else loss
#
# 					# Compute gradients
# 					loss.backward()
#
# 					# Perform optimization step after accumulating gradients for self.config.grad_acc_steps steps
# 					if (step + 1) % self.config.grad_acc_steps == 0:
# 						torch.nn.utils.clip_grad_norm_(self.pw_model.parameters(), self.config.max_grad_norm)
# 						self.optimizer.step()
# 						self.scheduler.step()
# 						self.optimizer.zero_grad()
#
# 					# Eval on eval dataset
# 					if (step + 1) % (self.config.eval_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info("Evaluation on the development dataset")
# 						self.pw_model.eval()
# 						eval_res = self.evaluate(pw_model=self.pw_model, eval_dataloader=self.dev_dataloader,
# 												 first_segment_end=self.config.max_input_len)
# 						wandb.log({"dev/epoch": epoch_iter,
# 								   "dev/step": epoch_iter*len(total_paths) + step,
# 								   "dev/loss": eval_res["loss"]})
# 						self.pw_model.train()
#
# 						if best_dev_loss is None or eval_res["loss"] < best_dev_loss:
# 							LOGGER.info("***** Saving best model on dev set *****")
# 							best_dev_loss = eval_res["loss"]
# 							wandb.run.summary["best_dev_loss"] = best_dev_loss
# 							self.save_model(res_dir=os.path.join(self.config.model_dir, "best_wrt_dev"))
#
# 					# print loss metrics
# 					if (step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
# 						LOGGER.info(
# 							f"Step {step} - epoch {epoch_iter}"
# 							f" average loss: {avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)}\n"
# 						)
# 						avg_train_loss = avg_train_loss / (self.config.print_interval * self.config.grad_acc_steps)
# 						wandb.log({"train/epoch": epoch_iter,
# 								   "train/step": epoch_iter*total_paths + step,
# 								   "train/step_frac": float(step)/total_paths,
# 								   "train/loss": avg_train_loss})
# 						avg_train_loss = 0.
#
#
# 			LOGGER.info("***** Saving fine - tuned model *****")
# 			self.save_model(res_dir=os.path.join(self.config.model_dir, "curr_epoch"))
#
# 		except Exception as e:
# 			embed()
# 			raise e
#
#
# 	@staticmethod
# 	def evaluate(pw_model, eval_dataloader, **kwargs):
# 		"""
# 		Compute loss with given pairwise model and data
# 		:param pw_model:
# 		:param eval_dataloader:
# 		:return:
# 		"""
# 		pw_model.eval()
# 		eval_loss_list = []
#
# 		for step, batch in enumerate(eval_dataloader):
#
# 			batch_pos_pairs, batch_neg_pairs = batch
#
# 			batch_pos_pairs = batch_pos_pairs.to(pw_model.device)
# 			batch_neg_pairs = batch_neg_pairs.to(pw_model.device)
#
# 			with torch.no_grad():
# 				loss = pw_model(batch_pos_pairs, batch_neg_pairs, first_segment_end=kwargs["first_segment_end"])
# 				eval_loss_list += [loss.item()]
#
# 		eval_loss = np.mean(eval_loss_list)
# 		return {"loss": eval_loss}