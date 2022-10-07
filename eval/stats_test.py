import sys
import numpy as np
from scipy import stats


# Code from https://github.com/rtmdrr/testSignificanceNLP

### Normality Check
# H0: data is normally distributed
def normality_check(data_A, data_B, name, alpha):

	if(name=="Shapiro-Wilk"):
		# Shapiro-Wilk: Perform the Shapiro-Wilk test for normality.
		shapiro_results = stats.shapiro([a - b for a, b in zip(data_A, data_B)])
		return shapiro_results[1]

	elif(name=="Anderson-Darling"):
		# Anderson-Darling: Anderson-Darling test for data coming from a particular distribution
		anderson_results = stats.anderson([a - b for a, b in zip(data_A, data_B)], 'norm')
		sig_level = 2
		if(float(alpha) <= 0.01):
			sig_level = 4
		elif(float(alpha)>0.01 and float(alpha)<=0.025):
			sig_level = 3
		elif(float(alpha)>0.025 and float(alpha)<=0.05):
			sig_level = 2
		elif(float(alpha)>0.05 and float(alpha)<=0.1):
			sig_level = 1
		else:
			sig_level = 0

		return anderson_results[1][sig_level]

	else:
		# Kolmogorov-Smirnov: Perform the Kolmogorov-Smirnov test for goodness of fit.
		ks_results = stats.kstest([a - b for a, b in zip(data_A, data_B)], 'norm')
		return ks_results[1]

## McNemar test
def calculateContingency(data_A, data_B, n):
	ABrr = 0
	ABrw = 0
	ABwr = 0
	ABww = 0
	for i in range(0,n):
		if data_A[i] == 1 and data_B[i] == 1:
			ABrr = ABrr+1
		if data_A[i] == 1 and data_B[i] == 0:
			ABrw = ABrw + 1
		if data_A[i] == 0 and data_B[i] == 1:
			ABwr = ABwr + 1
		else:
			ABww = ABww + 1
	return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
	statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
	pval = 1-stats.chi2.cdf(statistic,1)
	return pval


#Permutation-randomization
#Repeat R times: randomly flip each m_i(A),m_i(B) between A and B with probability 0.5, calculate delta(A,B).
# let r be the number of times that delta(A,B)<orig_delta(A,B)
# significance level: (r+1)/(R+1)
# Assume that larger value (metric) is better
def rand_permutation(data_A, data_B, n, R):
	rng = np.random.default_rng(0)
	delta_orig = float(sum([ x - y for x, y in zip(data_A, data_B)]))/n
	r = 0
	for x in range(0, R):
		temp_A = data_A
		temp_B = data_B
		samples = [rng.integers(1, 3) for i in range(n)] #which samples to swap without repetitions
		swap_ind = [i for i, val in enumerate(samples) if val == 1]
		for ind in swap_ind:
			temp_B[ind], temp_A[ind] = temp_A[ind], temp_B[ind]
		delta = float(sum([ x - y for x, y in zip(temp_A, temp_B)]))/n
		if delta <= delta_orig:
			r = r+1
	pval = float(r+1.0)/(R+1.0)
	return pval


#Bootstrap
#Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012),
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A, data_B, n, R):
	rng = np.random.default_rng(0)
	delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
	r = 0
	for x in range(0, R):
		temp_A = []
		temp_B = []
		samples = rng.integers(0,n,n) #which samples to add to the subsample with repetitions
		for samp in samples:
			temp_A.append(data_A[samp])
			temp_B.append(data_B[samp])
		delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
		if delta > 2*delta_orig:
			r = r + 1
	pval = float(r)/R
	return pval




def run_statistical_test_w_files(filename_A, filename_B, alpha, test_name):
	"""
	Run statistical test w/ given parameters
	:param filename_A: File w/ result for algo/model A. Each line contains eval measure on a sample/example/datapoint
	:param filename_B: File w/ result for algo/model B. Each line contains eval measure on a sample/example/datapoint
	:param alpha: Significance level threshold
	:param test_name: Statistical test to perform
	:return: pvalue for statistical test and pval for normality test
	"""

	with open(filename_A) as f:
		data_A = f.read().splitlines()

	with open(filename_B) as f:
		data_B = f.read().splitlines()

	data_A = list(map(float,data_A))
	data_B = list(map(float,data_B))

	return run_statistical_test(
		data_A=data_A,
		data_B=data_B,
		alpha=alpha,
		test_name=test_name
	)


def run_statistical_test(data_A, data_B, alpha, test_name):
	"""
	
	:param data_A: Array containing eval measure on each sample/example/datapoint for algo/model A
	:param data_B: Array containing eval measure on each sample/example/datapoint for algo/model B
	:param alpha: Significance level threshold
	:param test_name: Statistical test to perform
	:return: pvalue for statistical test and pval for normality test
	"""
	# print("\nPossible statistical tests: t-test, Wilcoxon, McNemar, Permutation, Bootstrap")
	# test_name = input("\nEnter name of statistical test: ")
	valid_tests = ["t-test", "Wilcoxon", "McNemar", "Permutation", "Bootstrap"]
	assert test_name in valid_tests, f"test_name={test_name} not in valid_tests={valid_tests}"

	### Normality Check
	# name== "Shapiro-Wilk" or name=="Anderson-Darling" or name=="Kolmogorov-Smirnov"
	normality_pval = normality_check(data_A, data_B, "Shapiro-Wilk", alpha)

	### Statistical tests
	if test_name == "t-test": # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b. for one sided test we multiply p-value by half
		t_results = stats.ttest_rel(data_A, data_B)
		pval = float(t_results[1])
		# # correct for one sided test
		# pval = pval / 2
		
	elif test_name == "Wilcoxon": # Wilcoxon: Calculate the Wilcoxon signed-rank test.
		wilcoxon_results = stats.wilcoxon(data_A, data_B)
		pval = wilcoxon_results[1]
		
	elif test_name == "McNemar" :
		print("\nThis test requires the results to be binary : A[1, 0, 0, 1, ...], B[1, 0, 1, 1, ...] for success or failure on the i-th example.")
		f_obs = calculateContingency(data_A, data_B, len(data_A))
		mcnemar_results = mcNemar(f_obs)
		pval = mcnemar_results
		
	elif test_name == "Permutation":
		R = max(10000, int(len(data_A) * (1 / float(alpha))))
		pval = rand_permutation(data_A, data_B, len(data_A), R)
		
	elif test_name == "Bootstrap":
		R = max(10000, int(len(data_A) * (1 / float(alpha))))
		pval = Bootstrap(data_A, data_B, len(data_A), R)
	else:
		raise NotImplementedError(f"Invalid name of statistical test = {test_name}")
	
	return pval, normality_pval

def main():
	if len(sys.argv) < 3:
		print("You did not give enough arguments\n ")
		sys.exit(1)
	filename_A = sys.argv[1]
	filename_B = sys.argv[2]
	alpha = sys.argv[3]
	
	if len(sys.argv) < 4:
		print("\nPossible statistical tests: t-test, Wilcoxon, McNemar, Permutation, Bootstrap")
		test_name = input("\nEnter name of statistical test: ")
	else:
		test_name = sys.argv[4]
	run_statistical_test(filename_A, filename_B, alpha, test_name)
	

if __name__ == "__main__":
	main()










