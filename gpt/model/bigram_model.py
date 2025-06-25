import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_size = cfg["MODEL"]["embed_size"]
        self.vocab_size = cfg["DATA"]["vocab_size"]

        self.embedding = nn.Embedding(self.embed_size, self.embed_size)
        self.pos_embeddings = nn.Embedding(self.embed_size, 1)
        self.project = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x):
        out = self.embedding(x)  # (B, T, embed_size)
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
