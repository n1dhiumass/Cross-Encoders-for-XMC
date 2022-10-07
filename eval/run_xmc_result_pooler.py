import sys
import json
import logging
import numpy as np
from collections import defaultdict
from IPython import embed
import os
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_avg_perf(base_dir, all_dirs, split, split_short):
	
	try:
		all_res = {}
		for curr_res_dir in all_dirs:
			filename = f"{base_dir}/{curr_res_dir}/eval/{split}/{split_short}.json_eval_result.json"
			
			# if not os.path.isfile(filename):
			# 	filename = f"{base_dir}/{curr_res_dir}/eval/{split}/{split_short}.json_eval_result.json"
			try:
				with open(filename, "r") as reader:
					res = json.load(reader)
					all_res[curr_res_dir] = res
			except Exception as e:
				LOGGER.info(f"For dataset = {curr_res_dir}, Error raised {str(e)}")
				
		print_str = ""
		for curr_res_dir in all_res:
			p1 = "{:.2f}".format(100*all_res[curr_res_dir]['prec_at_k']['1'])
			p3 = "{:.2f}".format(100*all_res[curr_res_dir]['prec_at_k']['3'])
			p5 = "{:.2f}".format(100*all_res[curr_res_dir]['prec_at_k']['5'])
			print_str += f"{curr_res_dir} &\t {p1} &\t{p3} &\t{p5}\n"
			
		LOGGER.info(f"{split_short}\n{print_str}")
		return all_res
	except Exception as e:
		embed()


def main():
	# res_dir = "../../results/4_DomainTransfer"
	# res_dir = "../../results/4_Zeshel/d=ent_link/m=bi_enc_l=ce_s=1234_hard_negs_wo_dp_bs_32/eval"
	# res_dir = "../../results/4_Zeshel/d=ent_link/m=bi_enc_l=ce_s=1234_random_negs_wo_dp_bs_32/eval"
	base_dir = "../../results/6_XMC/d=xmc"
	
	dirs = ["m=bi_enc_l=ce_s=1234_w_10_hard_negs_10Ksteps",
			"m=bi_enc_l=ce_s=1234_w_10_hard_negs_1Ksteps_titanx",
			"m=bi_enc_l=ce_s=1234_w_in_batch_negs",
			"m=bi_enc_l=hinge_s=1234_w_10_hard_negs_10Ksteps",
			"m=bi_enc_l=hinge_s=1234_w_10_hard_negs_1Ksteps_titanx",
			"m=bi_enc_l=hinge_s=1234_w_in_batch_negs",
			"m=bi_enc_l=hinge_s=1234_w_random_negs"]
			# "m=cross_enc_l=hinge_s=1234_w_random_negs"]
	dirs.reverse()
	res = {}
	for split_short, split in [("trn", "trn"), ("tst","tst")]:
		avg_res = get_avg_perf(base_dir=base_dir, all_dirs=dirs, split=split, split_short=split_short)
		
		res[split] = avg_res
	
	with open(f"{base_dir}/comb_res.json", "w") as fout:
		json.dump(res, fout, indent=4)
		
			
		
if __name__ == "__main__":
	main()
