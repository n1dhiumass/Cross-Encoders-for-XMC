import os
import sys
import json
import time
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
import pickle
from collections import defaultdict

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def main(data_fname, out_dir):

	
	world_to_ments = defaultdict(list)
	with open(data_fname, "r") as reader:
		for line in reader:
			ment_dict  = json.loads(line.strip())
			
			world_to_ments[ment_dict["type"]] += [ment_dict]
			
	
	LOGGER.info("Writing mentions for each world separately")
	Path(out_dir).mkdir(exist_ok=True, parents=True)
	for world in world_to_ments:
		with open(f"{out_dir}/{world}_mentions.jsonl", "w") as writer:
			for ment in world_to_ments[world]:
				writer.write(json.dumps(ment) + "\n")
				
		
		
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Split zeshel data into separate folder/files wrt worlds')
	
	parser.add_argument("--data_fname", type=str, required=True, help="Data file with data from multiple worlds")
	parser.add_argument("--out_dir", type=str, required=True, help="Output dir")
	
	
	args = parser.parse_args()
	
	_data_fname = args.data_fname
	_out_dir = args.out_dir
	
	main(data_fname=_data_fname, out_dir=_out_dir)
