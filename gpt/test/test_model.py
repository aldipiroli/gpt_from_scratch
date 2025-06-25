import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile

import numpy as np
import torch
from dataset.tokenizer import WordLevelTokenizer
from model.bigram_model import BigramModel
from model.gpt_model import Attention, GPTModel, MultiHeadAttention, TransformerLayer
from utils.misc import get_logger, load_config

with tempfile.TemporaryDirectory() as tmp_dir:
    logger = get_logger(tmp_dir)
    cfg = load_config("config/gpt_config.yaml")
    cfg["DATA"]["vocab_size"] = 65


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
    module = Attention(in_embed_size=embed_size, out_embed_size=embed_size)
    out = module(x)
    assert out.shape == (B, T, embed_size)


def test_multihead_attention():
    B, T, embed_size = 2, 10, 32
    num_heads = 4
    x = torch.randn(B, T, embed_size)
    module = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads)
    out = module(x)
    assert out.shape == (B, T, embed_size)
    assert (out == out).all()  # Check for NaN


def test_transformer_layer():
    B, T, embed_size = 2, 10, 32
    num_heads = 4
    x = torch.randn(B, T, embed_size)
    module = TransformerLayer(embed_size=embed_size, num_heads=num_heads)
    out = module(x)
    assert out.shape == (B, T, embed_size)


def test_gpt_model():
    data = "hello, world! This is the next"
    vocab = sorted(list(set(data)))
    cfg["DATA"]["vocab_size"] = len(vocab)
    context_len = cfg["MODEL"]["context_len"]

    tok = WordLevelTokenizer(vocab)
    model = GPTModel(cfg)
    B = 1
    x = data[:context_len]
    T = len(x)
    x_enc = tok.encode(x)
    x_enc = torch.tensor(np.array(x_enc))
    x_enc = x_enc.reshape(B, -1)  # B,T
    out = model(x_enc)
    assert out.shape == (B, T, len(vocab))


if __name__ == "__main__":
    print("All test passed!")
