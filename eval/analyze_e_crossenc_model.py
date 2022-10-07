import os
import sys
import json
import wandb
import torch
import pickle
import logging
import argparse
import numpy as np

from tqdm import tqdm
from IPython import embed
from pathlib import Path
from sklearn import manifold

from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info
from eval.eval_utils import score_topk_preds, compute_label_embeddings

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_stats(array_of_vals):
	
	res = {
		"min": np.min(array_of_vals),
		"max": np.max(array_of_vals),
		"mean": np.mean(array_of_vals),
		"std": np.std(array_of_vals),
		"p1": np.percentile(array_of_vals, 1),
		"p10": np.percentile(array_of_vals, 10),
		"p50": np.percentile(array_of_vals, 50),
		"p90": np.percentile(array_of_vals, 90),
		"p99": np.percentile(array_of_vals, 99),
	}
	res = {k:float(np.format_float_positional(v, 4)) for k,v in res.items()}
	return res


def avg_scores(list_of_score_dicts):
	try:
		metrics = {metric for score_dict in list_of_score_dicts for metric in score_dict}
		
		avg_scores = {}
		for metric in metrics:
			avg_scores[metric] = "{:.2f}".format(np.mean([float(score_dict[metric]) for score_dict in list_of_score_dicts]))
		
		return avg_scores
	except Exception as e:
		LOGGER.info("Exception raised in avg_scores")
		embed()
		raise e
	
	
def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	return {"indices":indices, "scores":scores}
	

def run_exact_inf(all_input_embeds, all_label_embeds, gt_labels, top_k, arg_dict, dataset_name, res_dir, misc):
	try:
		LOGGER.info("Running exact inference")
		res = {"arg_dict": arg_dict}
		n_ments, n_ents, embed_dim = all_input_embeds.shape
		
		sim_method_vals = ["ip", "cos"]
		for sim_method in sim_method_vals:
			if sim_method == "ip":
				crossenc_ment_to_ent_scores = torch.sum(all_input_embeds*all_label_embeds, dim=-1)
			elif sim_method == "cos":
				crossenc_ment_to_ent_scores = torch.nn.CosineSimilarity(dim=-1)(all_input_embeds, all_label_embeds)
			# elif sim_method == "cos2":
			# 	crossenc_ment_to_ent_scores = torch.sum(normalize(all_input_embeds, dim=-1, p=2)*normalize(all_label_embeds, dim=-1, p=2), dim=-1)
			else:
				raise NotImplementedError(f"Similarity method = {sim_method} not implemented")
			
			LOGGER.info(f"{sim_method} crossenc_ment_to_ent_scores shape {crossenc_ment_to_ent_scores.shape}")
			
			crossenc_top_k_scores, crossenc_top_k_indices = crossenc_ment_to_ent_scores.topk(top_k)
			crossenc_topk_preds = [(crossenc_top_k_indices, crossenc_top_k_scores)]
			crossenc_topk_preds = _get_indices_scores(crossenc_topk_preds)
			
			
			res[f"crossenc_{sim_method}"] = score_topk_preds(
				gt_labels=gt_labels,
				topk_preds=crossenc_topk_preds
			)
		
		curr_res_dir = f"{res_dir}/{dataset_name}/exact_crossenc_w_embeds/m={n_ments}_k={top_k}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
	except Exception as e:
		LOGGER.info("Exception raised in run_exact_inf")
		embed()
		raise e
	

def get_std_across_row_and_col_stats(all_embeds):
	"""
	Get stats for variation of embedding in all_embeds array along rows and along columns
	:param all_embeds: Tensor of shape: (n_ments, n_ents, embed_dim)
	:return:
	"""
	
	n_ments, n_ents, embed_dim = all_embeds.shape
	
	################################ ESTIMATE VARIATION ALONG COL (dim=0) DIMENSION ########################################
	# Average row embedding - averaged over all labels dimension
	avg_row_embeds = torch.mean(all_embeds, dim=1) # Shape: (n_ments, embed_dim)
	
	# [i,j] stores similarity b/w (i,j)th embedding and avg embedding of ith row
	# (n_ments, n_ents, 1) = (n_ments, n_ents, embed_dim) x (n_ments, embed_dim, 1)
	sim_w_avg_row_embed = torch.bmm(all_embeds, avg_row_embeds.unsqueeze(-1)) # Shape: (n_ments, n_ents, 1)
	sim_w_avg_row_embed = sim_w_avg_row_embed.squeeze(-1) # Shape: (n_ments, n_ents)
	
	# Find std dev of score for each embed - std-dev along col dimension=1, so axis = 1 (label axis)
	sim_w_avg_row_embed_std = np.std(sim_w_avg_row_embed.cpu().numpy(), axis=1) # shape: (n_ments,)
	sim_w_avg_row_embed_std_stats = get_stats(sim_w_avg_row_embed_std)
	
	
	################################ ESTIMATE VARIATION ALONG ROW (dim=0) DIMENSION ####################################
	# Average embedding in each column - averaged along input/mention dimension
	avg_col_embeds = torch.mean(all_embeds, dim=0, keepdim=True) # Shape: (1, n_ents, embed_dim)
	# (n_ments, n_ents, embed_dim) = (n_ments, n_ents, embed_dim) x (1, n_ents, embed_dim)
	sim_w_avg_col_embed = all_embeds * avg_col_embeds
	
	# i,j -> stores similarity b/w (i,j)th embedding and avg embedding of jth col (label)
	sim_w_avg_col_embed = torch.sum(sim_w_avg_col_embed, dim=-1) # shape: (n_ments, n_ents)
	
	# Find std dev of score for each embed - std-dev along row dimension=0, so axis = 0 (mention axis)
	sim_w_avg_col_embed_std = np.std(sim_w_avg_col_embed.cpu().numpy(), axis=0) # Shape : (n_ents, )
	sim_w_avg_col_embed_std_stats = get_stats(sim_w_avg_col_embed_std)
	
	
	################################ ESTIMATE VARIATION ALONG BOTH ROW AND COL (dim=0,1) DIMENSION #####################
	# Average embedding in each column - averaged along input/mention dimension
	avg_embeds = torch.mean(torch.mean(all_embeds, dim=0), dim=0) # Shape: (embed_dim, )
	LOGGER.info(f"avg_embeds.shape {avg_embeds.shape}")
	# (n_ments, n_ents, embed_dim) = (n_ments, n_ents, embed_dim) x (embed_dim, )
	sim_w_avg_embed = all_embeds * avg_embeds
	sim_w_avg_embed = torch.sum(sim_w_avg_embed, dim=-1) # shape: (n_ments, n_ents)
	LOGGER.info(f"Overall sim w avg embed shape :{sim_w_avg_embed.shape} {n_ments,n_ents}")
	
	sim_w_avg_embed = sim_w_avg_embed.view(-1).cpu().numpy() # Shape : (n_ments*n_ents,)
	LOGGER.info(f"Overall sim w avg embed shape :{sim_w_avg_embed.shape} {n_ments*n_ents}")
	overall_var_stats = get_stats(sim_w_avg_embed)
	
	# LOGGER.info(f"Variation along row dimension {sim_w_avg_row_embed_std_stats}")
	# LOGGER.info(f"Variation along col dimension {sim_w_avg_col_embed_std_stats}")
	res = {
		"row_variation": sim_w_avg_row_embed_std_stats,
		"col_variation": sim_w_avg_col_embed_std_stats,
		"overall_variation":overall_var_stats,
	}
	return res
	

def run_embed_variance_analysis(all_input_embeds, all_label_embeds, gt_labels, top_k, arg_dict, dataset_name, res_dir, misc):
	
	try:
		n_ments, n_ents, embed_dim = all_input_embeds.shape
		# 1. See variance, norm of embedding of an contextualized embedding of an input with all labels - and then average over all labels
		
		# See variance of mention embeddings across different mentions - this is to check if mention embedding is
		# roughly constant for all mentions and is it just entity embedding that is varying?
		# This can also be in-part see with t-SNE plots
		
		# 2. See variance in contextualized embedding of a label with all inputs - and then average over all labels
		# See variance of entity embeddings across different entities -
		# This can also be in-part see with t-SNE plots?
		
		res = {"arg_dict": arg_dict}
		LOGGER.info("\nVariation for mention embeddings")
		res["mention"] = get_std_across_row_and_col_stats(all_embeds=all_input_embeds)
		
		LOGGER.info("\nVariation for entity embeddings")
		res["entity"] = get_std_across_row_and_col_stats(all_embeds=all_label_embeds)
	
		
		curr_res_dir = f"{res_dir}/{dataset_name}/exact_crossenc_w_embeds/m={n_ments}_k={top_k}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		with open(f"{curr_res_dir}/embed_variation.json", "w") as fout:
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		# Plot t-SNE embeddings for all mention embeddings and all label embeddings -
		# tsne = manifold.TSNE(n_components=2, random_state=42)
		# mnist_tr = tsne.fit_transform(all_input_embeds)
		# Plot using scatter plot and colour embeddings by mention-id -
		# Maybe restrict number of mentions because plotting 100 colors would be too much - do not plot legend
		
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)} in run_embed_variance_analysis")
		embed()
		raise e


def run(dataset_name, data_fname, top_k, res_dir, misc,  arg_dict):
	try:
		assert top_k > 1
		
		##################################### Read precomputed data ####################################################
		LOGGER.info("Loading precomputed ment_to_ent embeds")
		with open(data_fname["crossenc_ment_and_ent_embeds"], "rb") as fin:
			dump_dict = pickle.load(fin)

			all_label_embeds = dump_dict["all_label_embeds"]
			all_input_embeds = dump_dict["all_input_embeds"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			entity_tokens_list = dump_dict["entity_tokens_list"]
		
		LOGGER.info("Finished loading")
		################################################################################################################
		
		n_ments, n_ents, embed_dim = all_input_embeds.shape
	
		# Map entity ids to local ids
		ent_id_to_local_id = {ent_id:idx for idx, ent_id in enumerate(entity_id_list)}
		gt_labels = np.array([ent_id_to_local_id[mention["label_id"]] for mention in test_data], dtype=np.int64)
	
		# run_exact_inf(
		# 	all_input_embeds=all_input_embeds,
		# 	all_label_embeds=all_label_embeds,
		# 	gt_labels=gt_labels,
		# 	top_k=top_k,
		# 	arg_dict=arg_dict,
		# 	dataset_name=dataset_name,
		# 	res_dir=res_dir,
		# 	misc=misc
		# )
		run_embed_variance_analysis(
			all_input_embeds=all_input_embeds,
			all_label_embeds=all_label_embeds,
			gt_labels=gt_labels,
			top_k=top_k,
			arg_dict=arg_dict,
			dataset_name=dataset_name,
			res_dir=res_dir,
			misc=misc
		)
		
		
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


 	
		
def main():
	data_dir = "../../data/zeshel"
	
	worlds = get_zeshel_world_info()

	parser = argparse.ArgumentParser( description='Use precomputed embeddings from crossencoder-w-embeds model for analysis')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir with precomputed score mats and dir to save results")
	parser.add_argument("--top_k", type=int, default=100, help="top-k entities to recall wrt crossencoder scores")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	
	args = parser.parse_args()
	data_name = args.data_name
	res_dir = args.res_dir
	top_k = args.top_k
	misc = args.misc
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds)
	
	run(
		dataset_name=data_name,
		data_fname=DATASETS[data_name],
		top_k=top_k,
		res_dir=res_dir,
		misc=misc,
		arg_dict=args.__dict__
	)

	
if __name__ == "__main__":
	main()

