import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile

import numpy as np
import torch
from dataset.tokenizer import WordLevelTokenizer
from model.bigram_model import BigramModel
from model.gpt_model import Attention, GPTModel
from utils.misc import get_logger, load_config

with tempfile.TemporaryDirectory() as tmp_dir:
    logger = get_logger(tmp_dir)
    cfg = load_config("config/gpt_config.yaml")
    cfg["DATA"]["vocab_size"] = 65


def test_gpt_model():
    model = GPTModel(cfg)
    assert True


def test_bigram_model():
    cfg_model = cfg["MODEL"]
    model = BigramModel(cfg)
    B, T = 2, cfg_model["context_len"]
    x = torch.randint(0, cfg["MODEL"]["context_len"], (B, T))
    out = model(x)
    assert out.shape == (B, T, cfg["DATA"]["vocab_size"])


def test_bigram_model_generation():
    x = "hello, world! This is the next"
    vocab = sorted(list(set(x)))
    cfg["DATA"]["vocab_size"] = len(vocab)

    tok = WordLevelTokenizer(vocab)
    model = BigramModel(cfg)
    B = 1
    x_enc = tok.encode(x)
    x_enc = torch.tensor(np.array(x_enc))
    x_enc = x_enc.reshape(B, -1)  # B,T
    max_tokens = 10
    out = model.generate(x_enc, max_tokens=max_tokens)
    assert out.shape == (B, x_enc.shape[1] + max_tokens)


def test_attention():
    embed_size = 32
    B, T = 2, 10
    x = torch.randn(B, T, embed_size)
    module = Attention(embed_size=embed_size)
    out = module(x)
    assert out.shape == (B, T, embed_size)


if __name__ == "__main__":
    print("All test passed!")
