import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile

import torch
from model.bigram_model import BigramModel
from model.gpt_model import GPTModel
from utils.misc import get_logger, load_config

with tempfile.TemporaryDirectory() as tmp_dir:
    logger = get_logger(tmp_dir)
    cfg = load_config("config/gpt_config.yaml")


def test_gpt_model():
    model = GPTModel(cfg)
    assert True


def test_bigram_model():
    cfg_model = cfg["MODEL"]
    model = BigramModel(cfg)
    B, T = 2, cfg_model["context_len"]
    x = torch.randint(0, cfg["MODEL"]["context_len"], (B, T))
    out = model(x)
    assert out.shape == (B, T, cfg_model["embed_size"])


if __name__ == "__main__":
    print("All test passed!")
