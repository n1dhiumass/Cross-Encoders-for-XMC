import sys
import json
import torch
import argparse

import logging
from IPython import embed
from pathlib import Path

from blink.biencoder.biencoder import load_biencoder
from blink.crossencoder.crossencoder import load_crossencoder

from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def port_cross_encoder(crossencoder_config, crossencoder_model_file, res_dir):
	
	# load crossencoder model
	with open(crossencoder_config) as json_file:
		crossencoder_params = json.load(json_file)
		crossencoder_params["path_to_model"] = crossencoder_model_file
	blink_crossencoder = load_crossencoder(crossencoder_params)


	init_crossenc_model_config = "config/port_blink_models/el_cross_enc.json"
	with open(init_crossenc_model_config, "r") as fin:
		config = json.load(fin)
		my_crossencoder = CrossEncoderWrapper.load_model(config=config)
		
	assert isinstance(my_crossencoder, CrossEncoderWrapper)
	# my_crossencoder.model.encoder.bert_model = blink_crossencoder.model.encoder.bert_model.state_dict()
	
	LOGGER.info("Now transferring encoder model params from BLINK models to my model class object")
	encoder_state_dict = blink_crossencoder.model.encoder.state_dict()
	
	my_crossencoder.model.encoder.load_state_dict(encoder_state_dict)
	my_crossencoder.model.bert_config = blink_crossencoder.model.config
	
	LOGGER.info("Saving model")
	my_crossencoder.save_model(res_dir=f"{res_dir}/crossencoder")
	
	with open(f"{res_dir}/crossencoder/wrapper_config.json", "r") as fin:
		config = json.load(fin)
		my_crossencoder_reload = CrossEncoderWrapper.load_model(config=config)
		
	


def port_bi_encoder(biencoder_config, biencoder_model_file, res_dir):
	
	try:
		with open(biencoder_config) as json_file:
			biencoder_params = json.load(json_file)
			biencoder_params["path_to_model"] = biencoder_model_file
		blink_biencoder = load_biencoder(biencoder_params)
		
		
		init_bienc_model_config = "config/port_blink_models/el_bi_enc.json"
		
		
		with open(init_bienc_model_config, "r") as fin:
			config = json.load(fin)
			my_biencoder = BiEncoderWrapper.load_model(config=config)
			
		# my_crossencoder.model.encoder.bert_model = blink_crossencoder.model.encoder.bert_model.state_dict()
		
		LOGGER.info("Now transferring encoder model params from BLINK models to my model class object")
		input_encoder_state_dict = blink_biencoder.model.context_encoder.state_dict()
		label_encoder_state_dict = blink_biencoder.model.cand_encoder.state_dict()
		
		my_biencoder.model.input_encoder.load_state_dict(input_encoder_state_dict)
		my_biencoder.model.label_encoder.load_state_dict(label_encoder_state_dict)
		my_biencoder.model.bert_config = blink_biencoder.model.config
		
		LOGGER.info("Saving model")
		
		my_biencoder.save_model(res_dir=f"{res_dir}/biencoder")
		
		with open(f"{res_dir}/biencoder/wrapper_config.json", "r") as fin:
			config = json.load(fin)
			my_biencoder_reload = BiEncoderWrapper.load_model(config=config)
		
		debug_bi(blink_biencoder, my_biencoder, my_biencoder_reload)
		LOGGER.info("Finished successfully...")
		embed()
	except Exception as e:
		embed()
		raise e

		
def debug_bi(bienc_1, bienc_2, bienc_3):
	try:
		f = [  101, 10930,  2850, 11906,  1000,  1012, 19962, 22599,  1012,  1996, 2792,  4269,  2073,     1,  1996,  3025,  2792,     2,  2187,  2125, 1012,  1996,  8529, 20709,  2078,  1049, 16257,  2003,  7866, 10930, 2850,   102,  6070,  2683,  2475, 14255, 16555,  2050, 21080,     3, 6070,  2683,  2475, 14255, 16555,  2050, 21080,  6070,  2683,  2475, 14255, 16555,  2050, 21080,  2003,  1037, 16012,  8713,  2571,  2275, 2207,  1999,  2294,  1012,  2009,  3397,  1037,  3481,  2109,  2011, 1996, 14255, 16555,  2050,  1998,  2176,  3929,  2396,  2594,  7068, 3468,  4481,  1997, 14855, 10820,  1010,  4290,  2226,  1010, 27793, 2243,  1998,  9027,  4817,  1012,  5614, 10234,  1996,  3481,  2003, 1037,  3335,  2571, 22742,  1012,  1996,  3481,  2036,  3397,  1037, 7173,  1012,  1996,  2000,  2050,  1999,  7556,  4536,  1037, 21713, 16555,  2243, 25645,  6804,  2007,  1037, 23564,  5302,  2099, 10336, 22742,  1012,   102,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0]
	
		bienc_1.eval()
		bienc_2.eval()
		bienc_3.eval()
		
		tokens = torch.LongTensor(f).to(bienc_1.device).unsqueeze(0)
		score1 = bienc_1.encode_candidate(tokens)
		score2 = bienc_2.encode_candidate(tokens)
		score3 = bienc_3.encode_candidate(tokens)
	
		print("Score 1", score1)
		print("Score 2", score2)
		print("Score 3", score3)
		embed()
	except Exception as e:
		embed()
		raise e
	
def main():
	try:
		pretrained_dir = "../../BLINK_models"
		res_dir = f"../../BLINK_models_ported"
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		
		if sys.argv[1] == "cross":
			port_cross_encoder(crossencoder_config=f"{pretrained_dir}/crossencoder_wiki_large.json",
							crossencoder_model_file=f"{pretrained_dir}/crossencoder_wiki_large.bin",
							res_dir=res_dir)
		elif sys.argv[1] == "bi":
			port_bi_encoder(biencoder_config=f"{pretrained_dir}/biencoder_wiki_large.json",
							biencoder_model_file=f"{pretrained_dir}/biencoder_wiki_large.bin",
							res_dir=res_dir)
		else:
			pass
		
	except Exception as e:
		embed()
		raise e
	
if __name__ == "__main__":
	main()
