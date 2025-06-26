import argparse

import torch
from dataset.tiny_shakespeare import Tinyshakespeare
from model.gpt_model import GPTModel
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


@torch.no_grad()
def run_inference(args):
    config = load_config(args.config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    val_dataset = Tinyshakespeare(cfg=config, mode="val", logger=logger)
    config["DATA"]["vocab_size"] = val_dataset.vocab_size
    trainer.set_dataset(val_dataset, val_dataset, data_config=config["DATA"])

    model = GPTModel(config)
    trainer.set_model(model)
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.load_checkpoint(args.ckpt)
    trainer.generate_output(args.prompt, num_gen_tokens=args.out_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gpt_config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_tokens", type=int, default=1000)
    args = parser.parse_args()
    run_inference(args)
