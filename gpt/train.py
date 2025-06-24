import argparse

from dataset.tiny_shakespeare import Tinyshakespeare
from model.gpt_model import GPTModel
from model.loss_function import GPTLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model = GPTModel(config)
    trainer.set_model(model)

    train_dataset = Tinyshakespeare(cfg=config, mode="train", logger=logger)
    val_dataset = Tinyshakespeare(cfg=config, mode="val", logger=logger)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(GPTLoss())

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/gpt_config.yaml", help="Config path")
    args = parser.parse_args()
    train(args)
