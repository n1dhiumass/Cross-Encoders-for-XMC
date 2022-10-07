import sys
import logging
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def split_data(train_file, dev_frac):
	"""
	Split data in train_file into train/dev dataset based on given dev fraction
	:param train_file:
	:param dev_frac:
	:return: pair of lists containing train and dev data.
	"""
	input_data = []
	LOGGER.info("Reading data")
	with open(train_file, "r") as reader:
		input_data = [line.strip() for line in tqdm(reader)]
	
	LOGGER.info("Created input_data array")
	n = len(input_data)
	n_train = int(np.ceil(n*(1 - dev_frac)))
	n_dev = n - n_train
	
	rng = np.random.default_rng(0) # Get randomness generator with a fixed seed
	all_indices  = np.arange(n)
	rng.shuffle(all_indices)
	
	LOGGER.info("Shuffled indices")
	train_indices = all_indices[:n_train]
	dev_indices = all_indices[n_train: n_train + n_dev]
	
	train_data  = [input_data[i] for i in train_indices]
	dev_data  = [input_data[i] for i in dev_indices]
	LOGGER.info(f"Length of data = {n}")
	LOGGER.info(f"Length of training data = {len(train_data)}")
	LOGGER.info(f"Length of dev data = {len(dev_data)}")
	return train_data, dev_data



def main():
	parser = argparse.ArgumentParser( description='Script to split XMC training data in train and validation data')
	
	parser.add_argument("--train_file", type=str, required=True, help="training data file")
	parser.add_argument("--dev_frac", type=float, required=True, help="fraction of data for dev dataset")
	parser.add_argument("--out_dir", type=str, required=True, help="output folder dir")
	
	args = parser.parse_args()
	
	train_file = args.train_file
	dev_frac = args.dev_frac
	out_dir = args.out_dir
	
	Path(out_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present
	
	train_data, dev_data = split_data(train_file=train_file, dev_frac=dev_frac)
	
	LOGGER.info("Writing training data")
	with open(f"{out_dir}/trn.json", "w") as writer:
		for line in tqdm(train_data):
			writer.write(line + "\n")
	
	LOGGER.info("Writing dev data")
	with open(f"{out_dir}/dev.json", "w") as writer:
		for line in tqdm(dev_data):
			writer.write(line + "\n")
	

if __name__ == '__main__':
	main()
