import sys
import logging
import numpy as np
import xclib.data.data_utils as data_utils
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.sparse import topk, binarize
from scipy.sparse import load_npz, save_npz
import scipy.sparse as sp
from IPython import embed

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

############## From pyxclib codebase - start ####################################

def _broad_cast(mat, like):
	if isinstance(like, np.ndarray):
		return np.asarray(mat)
	elif sp.issparse(mat):
		return mat
	else:
		raise NotImplementedError(
			"Unknown type; please pass csr_matrix, np.ndarray or dict.")


def _get_topk(X, pad_indx=0, k=5, sorted=False):
	"""
	Get top-k indices (row-wise); Support for
	* csr_matirx
	* 2 np.ndarray with indices and values
	* np.ndarray with indices or values
	"""
	if sp.issparse(X):
		X = X.tocsr()
		X.sort_indices()
		pad_indx = X.shape[1]
		indices = topk(X, k, pad_indx, 0, return_values=False)
	elif type(X) == np.ndarray:
		# indices are given
		assert X.shape[1] >= k, "Number of elements in X is < {}".format(k)
		if np.issubdtype(X.dtype, np.integer):
			assert sorted, "sorted must be true with indices"
			indices = X[:, :k] if X.shape[1] > k else X
		# values are given
		elif np.issubdtype(X.dtype, np.floating):
			_indices = np.argpartition(X, -k)[:, -k:]
			_scores = np.take_along_axis(
				X, _indices, axis=-1
			)
			indices = np.argsort(-_scores, axis=-1)
			indices = np.take_along_axis(_indices, indices, axis=1)
	elif type(X) == dict:
		indices = X['indices']
		scores = X['scores']
		assert compatible_shapes(indices, scores), \
			"Dimension mis-match: expected array of shape {} found {}".format(
				indices.shape, scores.shape)
		assert scores.shape[1] >= k, "Number of elements in X is < {}".format(
			k)
		# assumes indices are already sorted by the user
		if sorted:
			return indices[:, :k] if indices.shape[1] > k else indices

		# get top-k entried without sorting them
		if scores.shape[1] > k:
			_indices = np.argpartition(scores, -k)[:, -k:]
			_scores = np.take_along_axis(
				scores, _indices, axis=-1
			)
			# sort top-k entries
			__indices = np.argsort(-_scores, axis=-1)
			_indices = np.take_along_axis(_indices, __indices, axis=-1)
			indices = np.take_along_axis(indices, _indices, axis=-1)
		else:
			_indices = np.argsort(-scores, axis=-1)
			indices = np.take_along_axis(indices, _indices, axis=-1)
	else:
		raise NotImplementedError(
			"Unknown type; please pass csr_matrix, np.ndarray or dict.")
	return indices


def compatible_shapes(x, y):
	"""
	See if both matrices have same shape

	Works fine for the following combinations:
	* both are sparse
	* both are dense
	
	Will only compare rows when:
	* one is sparse/dense and other is dict
	* one is sparse and other is dense

	** User must ensure that predictions are of correct shape when a
	np.ndarray is passed with all predictions.
	"""
	# both are either sparse or dense
	if (sp.issparse(x) and sp.issparse(y)) \
		or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
		return x.shape == y.shape

	# compare #rows if one is sparse and other is dict or np.ndarray
	if not (isinstance(x, dict) or isinstance(y, dict)):
		return x.shape[0] == y.shape[0]
	else:
		if isinstance(x, dict):
			return len(x['indices']) == len(x['scores']) == y.shape[0]
		else:
			return len(y['indices']) == len(y['scores']) == x.shape[0]


def _setup_metric(X, true_labels, inv_psp=None, k=5, sorted=False):
	assert compatible_shapes(X, true_labels), \
		"ground truth and prediction matrices must have same shape."
	num_instances, num_labels = true_labels.shape
	indices = _get_topk(X, num_labels, k, sorted)
	ps_indices = None
	if inv_psp is not None:
		_mat = sp.spdiags(inv_psp, diags=0,
						  m=num_labels, n=num_labels)
		_psp_wtd = _broad_cast(_mat.dot(true_labels.T).T, true_labels)
		ps_indices = _get_topk(_psp_wtd, num_labels, k)
		inv_psp = np.hstack([inv_psp, np.zeros((1))])

	idx_dtype = true_labels.indices.dtype
	true_labels = sp.csr_matrix(
		(true_labels.data, true_labels.indices, true_labels.indptr),
		shape=(num_instances, num_labels+1), dtype=true_labels.dtype)

	# scipy won't respect the dtype of indices
	# may fail otherwise on really large datasets
	true_labels.indices = true_labels.indices.astype(idx_dtype)
	return indices, true_labels, ps_indices, inv_psp

def _eval_flags(indices, true_labels, inv_psp=None):
	if sp.issparse(true_labels):
		nr, nc = indices.shape
		rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
		eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
	elif type(true_labels) == np.ndarray:
		eval_flags = np.take_along_axis(true_labels,
										indices, axis=-1)
	if inv_psp is not None:
		eval_flags = np.multiply(inv_psp[indices], eval_flags)
	return eval_flags


############## From pyxclib codebase - end ####################################


def get_filter_map(fname):
	if fname is not None:
		return np.loadtxt(fname).astype(np.int)
	else:
		return None


def filter_predictions(pred, mapping):
	if mapping is not None and len(mapping) > 0:
		print("Filtering labels.")
		pred[mapping[:, 0], mapping[:, 1]] = 0
		pred.eliminate_zeros()
	return pred



def recall_new(pred_labels, true_labels, k):
	"""
	This recall number takes into account both total number of gt labels for a datapoint
	as well as total number of label retrieved, and takes a min of these two numbers
	when evaluating recall@k
	:param pred_labels:
	:param true_labels:
	:param k:
	:return:
	"""
	indices, true_labels, _, _ = _setup_metric(
		pred_labels, true_labels, k=k, sorted=sorted)
	
	deno_wrt_k = np.arange(1, k+1)

	# Find total number of gt label for each datapoint
	deno_wrt_num_pos_label = true_labels.sum(axis=1)
	deno_wrt_num_pos_label[deno_wrt_num_pos_label == 0] = 1

	# Final denominator should be min of (total_num_labels_for_datapoint, k)
	final_deno  = np.minimum(deno_wrt_k, deno_wrt_num_pos_label)
	
	eval_flags = _eval_flags(indices, true_labels, None)
	eval_flags = np.cumsum(eval_flags, axis=-1)
	recall = np.mean(np.divide(eval_flags, final_deno), axis=0)
	# embed()
	return np.ravel(recall)
	


	


def eval_predictions(gt_fname, pred_fname, filter_fname, k):
	
	
	true_labels = data_utils.read_sparse_file(gt_fname)
	pred_labels = load_npz(pred_fname)
	
	# Mapping from datapoint id to label ids which should be ignored during prediction
	mapping = get_filter_map(filter_fname) # This contains mapping for reciprocal pairs
	
	# Modifies pred_labels in-place. For each datapoint, it removes any label using the filtered labels mapping
	filtered_pred_labels = filter_predictions(load_npz(pred_fname), mapping)
	
	embed()
	acc = xc_metrics.Metrics(true_labels)
	
	LOGGER.info("\n\n")
	LOGGER.info(f"Recall (new) Result w/o filtering \n{recall_new(pred_labels, true_labels, k)}")
	LOGGER.info(f"Recall (new) Result w/ filtering \n{recall_new(filtered_pred_labels, true_labels, k)}")
	
	LOGGER.info("\n\n")
	LOGGER.info(f"Eval Result w/o filtering \n{xc_metrics.format(*acc.eval(pred_labels, k))}")
	LOGGER.info(f"Eval Result w filtering \n{xc_metrics.format(*acc.eval(filtered_pred_labels, k))}")
	
	LOGGER.info("\n\n")
	recall = xc_metrics.recall(pred_labels, true_labels, k=k)
	LOGGER.info(f"Recall Result w/o filtering \n{recall}")
	recall = xc_metrics.recall(filtered_pred_labels, true_labels, k=k)
	LOGGER.info(f"Recall Result w/ filtering \n{recall}")
	
	
	
	# embed()
	
	# from collections import defaultdict
	# # filter_dict = defaultdict(set)
	# filter_dict = {}
	# for x,y in mapping:
	# 	if x not in filter_dict:
	# 		filter_dict[x] = {y}
	# 	else:
	# 		filter_dict[x].add(y)
	#
	# absent_ctr = 0
	# present_ctr = 0
	# for x, reci_nbrs in filter_dict.items():
	# 	for y in reci_nbrs:
	# 		if y in filter_dict and x in filter_dict[y]:
	# 			present_ctr += 1
	# 		else:
	# 			absent_ctr += 1
	#
	# LOGGER.info(f"Present ctr = {present_ctr}")
	# LOGGER.info(f"Absent ctr = {absent_ctr}")
	
	pass
	


if __name__ == "__main__":
	
	data_for_inference = "tst"
	# data_for_inference = "trn"
	base_data_dir = "../../data/1_LF-AmazonTitles-131K/bow"
	base_res_dir = "../../results/SiameseXML/Astec/1_LF-AmazonTitles-131K/bow/v_default_params_0"
	
	test_fname = f"{base_data_dir}/{data_for_inference}_X_Y.txt"
	pred_fname = f"{base_res_dir}/{data_for_inference}_predictions_clf.npz"
	
	if data_for_inference == "tst":
		filter_fname  = f"{base_data_dir}/filter_labels_test.txt"
	elif data_for_inference == "trn":
		filter_fname  = f"{base_data_dir}/filter_labels_train.txt"
	else:
		raise NotImplementedError(f"data_for_inference = {data_for_inference} not supported")
	
	k = 10
	eval_predictions(
		pred_fname=pred_fname,
		gt_fname=test_fname,
		filter_fname=filter_fname,
		k=k
	)
	
	# train_fname = "../../data/1_LF-Amazon-131K/trn_X_Y.txt"
	# filter_fname  = "../../data/1_LF-Amazon-131K/filter_labels_train.txt"
	# pred_fname = "../../results/SiameseXML/Astec/1_LF-Amazon-131K/v_default_params_0/extreme/t_predictions_clf.npz"
	# read_test_predictions(
	# 	pred_fname=pred_fname,
	# 	gt_fname=train_fname,
	# 	filter_fname=filter_fname
	# )
	
	pass
