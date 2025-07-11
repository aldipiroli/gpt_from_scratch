import argparse

from dataset.finetune_dataset import FineTuneDataset
from model.bigram_model import BigramModel
from model.gpt_model import GPTModel
from model.loss_function import GPTLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer

__all__models__ = {"GPTModel": GPTModel, "BigramModel": BigramModel}


def train(args):
    config = load_config(args.config)
    logger, now = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger, now)

    train_dataset = FineTuneDataset(cfg=config, mode="train", logger=logger)
    val_dataset = FineTuneDataset(cfg=config, mode="val", logger=logger)
    config["DATA"]["vocab_size"] = train_dataset.vocab_size

    model = __all__models__[config["MODEL"]["model_name"]](config)
    trainer.set_model(model)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(GPTLoss())
    trainer.load_checkpoint(args.ckpt, skip_otimizer=True)
    trainer.fine_tune()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gpt_config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    train(args)
