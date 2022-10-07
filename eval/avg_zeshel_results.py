import sys
import json
import logging
import numpy as np
from collections import defaultdict
from IPython import embed

from eval.run_cross_encoder_w_binenc_retriever_zeshel import score_topk_preds
from utils.zeshel_utils import N_MENTS_ZESHEL

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def process_domain(res_dir, dataset_name, n_ment, top_k, batch_size):
	curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}"
	
	
	curr_gt_labels = json.load(open(f"{curr_res_dir}/gt_labels.txt", "r"))
	
	bienc_topk_preds = json.load(open(f"{curr_res_dir}/bienc_topk_preds.txt", "r"))
	crossenc_topk_preds_w_rand_retrvr = json.load(open(f"{curr_res_dir}/crossenc_topk_preds_w_rand_retrvr.txt", "r"))
	crossenc_topk_preds_w_bienc_retrvr = json.load(open(f"{curr_res_dir}/crossenc_topk_preds_w_bienc_retrvr.txt", "r"))
	
	res = {"bienc": score_topk_preds(gt_labels=curr_gt_labels,
									 topk_preds=bienc_topk_preds),
		   "crossenc_w_bienc_retrvr": score_topk_preds(gt_labels=curr_gt_labels,
													   topk_preds=crossenc_topk_preds_w_bienc_retrvr),
		   "crossenc_w_rand_retrvr": score_topk_preds(gt_labels=curr_gt_labels,
													  topk_preds=crossenc_topk_preds_w_rand_retrvr)}
	
	with open(f"{curr_res_dir}/res_detailed.json", "w") as fout:
		json.dump(res, fout)

	

def get_avg_perf(res_dir, datasets, n_ment, top_k, batch_size, misc, inf_type):
	
	try:
		all_res = {}
		for dataset_name in datasets:
			if inf_type == "exact":
				curr_res_file = f"{res_dir}/{dataset_name}/exact_crossenc/m={n_ment}_k={top_k}_{misc}/res.json"
			elif inf_type == "bi+cross":
				curr_res_file = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}/res.json"
			elif inf_type == "bi+cross_w_exact":
				curr_res_file = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}/res_w_approx_exact_inf.json"
			else:
				raise NotImplementedError(f"inf_type={inf_type} not supported")
			
			try:
				with open(f"{curr_res_file}", "r") as reader:
					res = json.load(reader)
					all_res[dataset_name] = res
					if "extra_info" in all_res[dataset_name]: del all_res[dataset_name]["extra_info"]
					if "arg_dict" in all_res[dataset_name]: del all_res[dataset_name]["arg_dict"]
					
			except Exception as e:
				LOGGER.info(f"For dataset = {dataset_name}, Error raised {str(e)}")
		
		models = {model for res in all_res.values() for model in res}
		metrics = {metric for res in all_res.values() for model_res in res.values() for metric in model_res}
		
		if "extra_info" in models: models.remove("extra_info")
		if "arg_dict" in models: models.remove("arg_dict")
		avg_res = defaultdict(dict)
		for model in sorted(models):
			for metric in sorted(metrics):
				a = [float(res[model][metric]) for dataset_name, res in all_res.items()]
				avg_res[model][metric] = np.mean(a)
				
				a = [N_MENTS_ZESHEL[dataset_name]*float(res[model][metric]) for dataset_name, res in all_res.items()]
				total = np.sum(N_MENTS_ZESHEL[dataset_name] for dataset_name in all_res)
				avg_res[model]["micro_"+metric] = np.sum(a)/total
		
		LOGGER.info(f"\n{json.dumps(avg_res, indent=4)}")
		return avg_res
	except Exception as e:
		embed()

	
def main():

	res_dir_vals = [
		# (
		# 	"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=precomp_s=1234_w_cls_w_lin_6_387/63_of_500_cur_negs_nm_500_ne_200/epoch_1",
		# 	["eoe-1-last.ckpt"],
		# ),
		(
			"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=precomp_s=1234_w_cls_w_lin_6_387/63_of_500_cur_negs_nm_500_ne_200_plus_500_bienc_negs/epoch_1",
			["eoe-1-last.ckpt"],
		),
		(
			"../../results/6_ReprCrossEnc/d=ent_link/hard_neg_training/cls_ce/m=cross_enc_l=ce_neg=precomp_s=1234_w_cls_w_lin_6_387/63_of_500_cur_negs_nm_500_ne_200_plus_100_bienc_negs/epoch_1",
			["eoe-1-last.ckpt"],
		),
	]
 
	
	n_ment = -1
	# n_ment = 100
	top_k = 64
	
	batch_size = 1
	# batch_size = 100
	# misc_vals = [""] # for exact inference
	
	
	# exact_model = False
	inf_type = "bi+cross"
	# inf_type = "bi+cross_w_exact"
	for res_dir, misc_vals in res_dir_vals:
		res_dir = res_dir + ("/score_mats" if inf_type == "exact" else "/eval")
		for misc in misc_vals:
			
			train_worlds =  ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling",
							 "starwars", "world_of_warcraft"]
			test_worlds = ["forgotten_realms", "lego", "star_trek", "yugioh"]
			valid_worlds = ["coronation_street", "elder_scrolls", "ice_hockey", "muppets"]
			
			res = {}
			# for world_type, iter_worlds in [("train",train_worlds), ("test", test_worlds), ("valid", valid_worlds)]:
			for world_type, iter_worlds in [("test", test_worlds), ("valid", valid_worlds)]:
			# for world_type, iter_worlds in [("test", test_worlds)]:
				avg_res = get_avg_perf(res_dir=res_dir, datasets=iter_worlds, n_ment=n_ment,
									   top_k=top_k, batch_size=batch_size, misc=misc, inf_type=inf_type)
				res[world_type] = avg_res
			
			if inf_type == "exact":
				with open(f"{res_dir}/exact_crossenc_avg_res_m={n_ment}_k={top_k}_{misc}.json", "w") as fout:
					json.dump(res, fout, indent=4)
			elif inf_type == "bi+cross":
				with open(f"{res_dir}/avg_res_m={n_ment}_k={top_k}_{batch_size}_{misc}.json", "w") as fout:
					json.dump(res, fout, indent=4)
			elif inf_type == "bi+cross_w_exact":
				with open(f"{res_dir}/avg_res_w_approx_exact_inf_m={n_ment}_k={top_k}_{batch_size}_{misc}.json", "w") as fout:
					json.dump(res, fout, indent=4)
			else:
				raise NotImplementedError(f"inf_type = {inf_type} not supported")
				
				
	
if __name__ == "__main__":
	main()
