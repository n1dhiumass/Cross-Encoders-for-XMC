import os
import sys
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_code(config, out_dir = "code"):
	code_dir = "{}/{}".format(config.result_dir, out_dir)
	command = "rsync -avi --exclude=__pycache__ --exclude=slurm*.out --exclude=*.ipynb " \
			  "--exclude=.ipynb_checkpoints --exclude=.gitignore ../cross-encoder-xmc/  {}/".format(code_dir)
	
	Path(code_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present
	os.system(command)
	command = "echo {}  > {}/command.txt".format(" ".join(sys.argv), code_dir)
	os.system(command)
	

def write_metrics(all_metrics, filename):
	
	with open(filename, "w") as f:
		
		fieldnames = sorted(list(all_metrics.keys()))
		writer = csv.DictWriter(f, fieldnames=fieldnames)

		writer.writeheader()
		max_len = max( [len(all_metrics[key]) for key in all_metrics] )
		
		for ctr in range(max_len):
			row = {key: all_metrics[key][ctr] if len(all_metrics[key]) > ctr else "" for key in all_metrics}
			writer.writerow(rowdict=row)
	

def read_metrics(filename):
	
	all_metrics = defaultdict(list)
	with open(filename, "r") as f:
	
		reader = csv.DictReader(f)
		for row in reader:
			for key in row:
				if row[key] != "":
					try:
						all_metrics[key].append(float(row[key]))
					except:
						
						all_metrics[key].append(row[key])
				
	return all_metrics
	

def plot_metrics(all_metrics, res_dir):
	
	Path(res_dir+"/metrics").mkdir(parents=True, exist_ok=True)
	for metric in all_metrics:
	
		plt.clf()
		Y  = all_metrics[metric]
		plt.plot(range(len(Y)), Y, marker='o', label=str(metric))
		
		plt.title("{} vs iterations".format(metric))
		plt.xlabel("Number of iterations")
		plt.ylabel("{}".format(metric))
		plt.legend()
		plt.savefig("{}/metrics/{}.png".format(res_dir, metric))
		plt.close()
	
	metric_groups = defaultdict(list)
	for metric in all_metrics:
		met_list = metric.split("~")
		if len(met_list) == 1: continue
		metric_groups[met_list[0]].append(metric)
	
	for group in metric_groups:
		plt.clf()
		for metric in metric_groups[group]:
			Y  = all_metrics[metric]
			plt.plot(range(len(Y)), Y, marker='o', label=str(metric))
			
		plt.title("{} vs iterations".format(group))
		plt.xlabel("Number of iterations")
		plt.ylabel("{}".format(group))
		plt.legend()
		plt.savefig("{}/metrics/{}.png".format(res_dir, group))
		plt.close()

