import os
import sys
import json
import copy
import argparse


import logging

import numpy as np
from eval.eval_utils import get_reci_rank
from eval.stats_test import run_statistical_test
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, N_MENTS_ZESHEL
from IPython import embed

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
def get_pred_eval_per_input(topk_preds, gt_labels):
	res = []
	for idx,curr_gt in enumerate(gt_labels):
		res += [get_reci_rank(gt=curr_gt,
							  preds=topk_preds["indices"][idx],
							  scores=topk_preds["scores"][idx])]
	
	res = np.array(res)
	res = (res == 1)
	res = res.astype(np.int64)
	
	return res



LOGGER = logging.getLogger(__name__)


def main():


	
	worlds = get_zeshel_world_info()

	parser = argparse.ArgumentParser( description='Run statistical tests on files with predicted topk entities')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["test", "dev", "train"], required=True, help="Dataset name or split")
	parser.add_argument("--pred_file_1_template", type=str, required=True, help="Predicted topk file for model 1")
	parser.add_argument("--pred_file_2_template", type=str, required=True, help="Predicted topk file for model 2")
	parser.add_argument("--gt_file_template", type=str, required=True, help="Ground truth label file")
	parser.add_argument("--test_name", type=str, default="t-test", help="Statistical test to perform")
	
	
	args = parser.parse_args()

	data_name_or_split = args.data_name
	pred_file_1_template = args.pred_file_1_template
	pred_file_2_template = args.pred_file_2_template
	
	gt_file_template = args.gt_file_template
	test_name = args.test_name
	
	if data_name_or_split in ["test", "dev", "train"]:
		data_list = [data_name for split, data_name in worlds if split == data_name_or_split]
	else:
		data_list = [data_name_or_split]
	
	all_acc_1 = []
	all_acc_2 = []
	num_correct_1 = 0.
	num_correct_2 = 0.
	
	all_eval_list_1 = []
	all_eval_list_2 = []
	for data_name in data_list:
		LOGGER.info(f"Running statistical tests for domain = {data_name}")
		gt_file = gt_file_template.format(data_name)
		pred_file_1 = pred_file_1_template.format(data_name)
		pred_file_2 = pred_file_2_template.format(data_name)
		
		with open(gt_file, "r") as fin:
			gt_labels = json.load(fin)
			
		with open(pred_file_1, "r") as fin:
			topk_preds_1 = json.load(fin)
		
		with open(pred_file_2, "r") as fin:
			topk_preds_2 = json.load(fin)
		
		eval_list_1 =  get_pred_eval_per_input(topk_preds=topk_preds_1, gt_labels=gt_labels)
		eval_list_2 =  get_pred_eval_per_input(topk_preds=topk_preds_2, gt_labels=gt_labels)
		
		num_correct_1 += np.sum(eval_list_1)
		num_correct_2 += np.sum(eval_list_2)
		
		total = len(eval_list_1)
		acc_1 = np.sum(eval_list_1)/total
		acc_2 = np.sum(eval_list_2)/total
		
		all_acc_1 += [acc_1]
		all_acc_2 += [acc_2]
		
		all_eval_list_1 += [eval_list_1]
		all_eval_list_2 += [eval_list_2]
		
		# eval_list_2 = copy.deepcopy(eval_list_1)
		# eval_list_2[10] = (1 + eval_list_2[10]) % 2
		
		stat_test_res = run_statistical_test(
			data_A=eval_list_1,
			data_B=eval_list_2,
			test_name=test_name,
			alpha=0.05
		)
		
		LOGGER.info(f"Test result: p-val for {test_name} = {stat_test_res[0]}")
		LOGGER.info(f"Test result: p-val for normality test = {stat_test_res[1]}\n\n")
		
		# embed()
		
		# from IPython import embed
		# embed()
		# stat_test_res_file = ""
		#
		# if os.path.isfile(stat_test_res_file):
		# 	with open(stat_test_res_file, "r") as fin:
		# 		stats_res = json.load(fin)
		# else:
		# 	stats_res = {}
		#
		#
		# stats_res["i"] = {
		# 	"test": test_name,
		# 	"pval": stat_test_res[0],
		# 	"normality_pval": stat_test_res[1],
		# 	"pred_file_1": pred_file_1,
		# 	"pred_file_2": pred_file_2,
		# 	"gt_file": gt_file,
		# }
		#
		# with open(stat_test_res_file, "w") as fout:
		# 	json.dump(stats_res, fout)
		#
		# pass
	
	
	LOGGER.info(f"Avg acc 1  = {np.mean(all_acc_1)}")
	LOGGER.info(f"Avg acc 2  = {np.mean(all_acc_2)}")
	
	Z = np.sum([N_MENTS_ZESHEL[data_name] for data_name in data_list])
	LOGGER.info(f"Avg acc 1  = {num_correct_1/Z}")
	LOGGER.info(f"Avg acc 2  = {num_correct_2/Z}")
	
	all_eval_list_1 = np.concatenate(all_eval_list_1)
	all_eval_list_2 = np.concatenate(all_eval_list_2)
	
	stat_test_res = run_statistical_test(
		data_A=all_eval_list_1,
		data_B=all_eval_list_2,
		test_name=test_name,
		alpha=0.05
	)
	LOGGER.info(f"Combined Test result: p-val for {test_name} = {stat_test_res[0]}")
	LOGGER.info(f"Combined Test result: p-val for normality test = {stat_test_res[1]}\n\n")


if __name__ == "__main__":
	main()
