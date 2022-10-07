import numpy as np
import torch


class BaseAprrox(object):
	
	def __init__(self):
		pass
	
	def build(self):
		pass
		
	def get(self, i, j):
		"""
		return approximate value of index (i,j)
		:param i:
		:param j:
		:return:
		"""
		raise NotImplementedError



class CURApprox(BaseAprrox):

	def __init__(self, rows, cols, row_idxs, col_idxs):
		
		print("\n\n\n\nThis is biased towards giving better approximations for new rows or cols -- think about it\n\n\n")
		super(CURApprox, self).__init__()
		# M :(n x m) = C U R : (n x kc) X (kc x kr) X (kr x m)
		
		self.n = cols.shape[0]
		self.m = rows.shape[1]
		
		self.row_idxs = row_idxs
		self.col_idxs = col_idxs

		self.C = cols # n x kc
		self.R = rows # kr x m
		
		assert self._is_sorted(self.row_idxs), "row_idxs should be sorted"
		assert self._is_sorted(self.col_idxs), "col_idxs should be sorted"
		
		assert len(row_idxs) == self.R.shape[0]
		assert len(col_idxs) == self.C.shape[1]
		
		intersect_mat = self.C[row_idxs, :] # kr x kc
		
		assert torch.eq(self.C[row_idxs, :], self.R[:, col_idxs]), "Invalid rows and cols as their intersection does not match"

		self.U = np.linalg.pinv(intersect_mat) # kc x kr

		self.latent_rows, self.latent_cols = self._build_latent_row_cols(C=self.C, U=self.U, R=self.R)
	
	@staticmethod
	def _is_sorted(idx_list):
		return all(i < j for i,j in zip(idx_list[:-1], idx_list[1:]))
		
	@staticmethod
	def _build_latent_row_cols(C, U, R):

		latent_rows = C @ U # n x kr
		latent_cols = R # kr x m
	
		return latent_rows, latent_cols

	def get_rows(self, row_idxs):
		
		# len(row_idxs) x m) =  (len(row_idxs) x kr) X ( kr, m))
		ans = self.latent_rows[row_idxs,:] @ self.latent_cols
		return ans
	
	def get_cols(self, col_idxs):
		# n x len(col_idxs) =  (n x kr) X (kr, len(col_idxs))
		ans = self.latent_rows @ self.latent_cols[:, col_idxs]
		return ans
	
	def get(self, row_idxs, col_idxs):
		
		# len(row_idxs) x len(col_idxs) =  (len(row_idxs) x kr) X ( kr, len(col_idxs))
		ans = self.latent_rows[row_idxs,:] @ self.latent_cols[:, col_idxs]
		return ans

	def get_complete_col(self, sparse_cols):
		"""
		Take values in cols corresponding to anchor row indices and return complete cols
		:param sparse_cols:
		:return:
		"""
		# (n x *) = (n x kr) X (kr x *)
		dense_cols = self.latent_rows @ sparse_cols
		return dense_cols

	def topk_in_col(self, sparse_cols, k):
		"""
		Return top-k indices in these col(s)
		:return:
		"""
		
		return torch.topk(self.get_complete_col(sparse_cols=sparse_cols), k, dim=1)
	
	
	def get_complete_row(self, sparse_rows):
		"""
		Take values in rows corresponding to anchor col indices and return complete rows
		:param sparse_cols:
		:return:
		"""
		# (* x m) = (* x kr) X (kr x m)
		dense_rows = sparse_rows @ self.latent_cols
		return dense_rows

	def topk_in_row(self, sparse_rows, k):
		"""
		Return top-k indices in these row(s)
		:return:
		"""
		return torch.topk(self.get_complete_row(sparse_rows=sparse_rows), k, dim=1)
		

	
		
		
if __name__ == "__main__":

	rng = np.random.default_rng(seed=0)
	
	n = 4
	m = 6
	M = torch.tensor(np.random.rand(4, 6))
	
	kr = 2
	kc = 3
	
	row_idxs = sorted(rng.choice(n, size=kr, replace=False))
	col_idxs = sorted(rng.choice(m, size=kc, replace=False))
	
	rows = M[row_idxs]
	cols = M[col_idxs]
	
	approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols)
	

# def CUR(similarity_matrix, k, eps=1e-3, delta=1e-14, return_type="error", same=False):
# 	"""
# 	implementation of Linear time CUR algorithm of Drineas2006 et. al.
# 	input:
# 	1. similarity matrix in R^{n,d}
# 	2. integers c, r, and k
# 	output:
# 	1. either C, U, R matrices
# 	or
# 	1. CU^+R
# 	or
# 	1. error = similarity matrix - CU^+R
# 	"""
# 	rng = np.random.default_rng(seed=0)
#
# 	# Choose a subset of mentions as anchors
# 	anchor_mention_idxs = rng.choice(np.arange(n_ment), size=k_ment)
#
# 	n, d = similarity_matrix.shape
# 	c = min(k,n)
# 	r = min(k,n)
# 	try:
# 		assert 1 <= c and c <= d
# 	except AssertionError as error:
# 		print("1 <= c <= m is not true")
# 	try:
# 		assert 1 <= r and r <= n
# 	except AssertionError as error:
# 		print("1 <= r <= n is not true")
# 	try:
# 		assert 1 <= k and k <= min(c, r)
# 	except AssertionError as error:
# 		print("1 <= k <= min(c,r)")
#
# 	# using uniform probability instead of row norms
# 	pj = np.ones(d).astype(float) / float(d)
# 	qi = np.ones(n).astype(float) / float(n)
#
# 	# choose samples
# 	samples_c = rng.choice(range(d), c, replace=False)
#
# 	samples_r = rng.choice(range(n), r, replace=False)
#
# 	# grab rows and columns and scale with respective probability
# 	samp_pj = pj[samples_c]
# 	samp_qi = qi[samples_r]
#
# 	C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj * c)
# 	rank_k_C = C
#
# 	# modification works only because we assume similarity matrix is symmetric
# 	R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi * r)
# 	R = R.T
# 	psi = C[samples_r, :].T / np.sqrt(samp_qi * r)
# 	psi = psi.T
#
# 	U = np.linalg.pinv(rank_k_C.T @ rank_k_C)
# 	# i chose not to compute rank k reduction of U
# 	U = U @ psi.T
# 	return (C @ U) @ R