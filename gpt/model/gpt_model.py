import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, use_mask=True):
        super().__init__()
        self.in_embed_size = in_embed_size
        self.out_embed_size = out_embed_size
        self.use_mask = use_mask
        self.keys = nn.Linear(in_embed_size, out_embed_size)
        self.query = nn.Linear(in_embed_size, out_embed_size)
        self.values = nn.Linear(in_embed_size, out_embed_size)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)  # B,T,C

        qk = q @ k.transpose(2, 1)  # B, T, T

        if self.use_mask:  # mask past tokens
            B = x.shape[0]
            tril = torch.tril(torch.ones(B, T, T))
            qk = qk.masked_fill(tril == 0, float("-inf"))
            attention = F.softmax(qk / self.out_embed_size**-0.5, -1)

        attention = attention @ v
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0
        self.heads = nn.ModuleList([Attention(embed_size, embed_size // num_heads) for _ in range(num_heads)])

    def forward(self, x):
        out = out = torch.cat([h(x) for h in self.heads], -1)
        return out


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_size = cfg["MODEL"]["embed_size"]
        self.vocab_size = cfg["DATA"]["vocab_size"]

        self.embedding = nn.Embedding(self.embed_size, self.embed_size)
        self.pos_embeddings = nn.Embedding(self.embed_size, 1)
        self.project = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x):
        x_emb = self.embedding(x)  # (B, T, embed_size)
        pos_embedding = self.pos_embeddings(x)
        out = out + pos_embedding
        out = self.project(out)  # (B, T, C)
        return out

    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            out = self(x[:, -1])  # (B, T, C)
            out_prob = F.softmax(out, -1)
            out_token = torch.multinomial(out_prob, 1).to(x.device)
            x = torch.cat([x, out_token], -1)
        return x


if __name__ == "__main__":
    B, T, embed_size = 2, 10, 32
    num_heads = 4
    x = torch.randn(B, T, embed_size)
    module = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads)
    out = module(x)
    assert out.shape == (B, T, embed_size)
