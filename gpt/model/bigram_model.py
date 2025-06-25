import torch.nn as nn


class BigramModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_size = cfg["MODEL"]["embed_size"]
        self.embedding = nn.Embedding(self.embed_size, self.embed_size)
        self.pos_embeddings = nn.Embedding(self.embed_size, 1)
        self.project = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, x):
        B, T = x.shape
        out = self.embedding(x)  # B, T, C
        pos_embedding = self.pos_embeddings(x)
        out = out + pos_embedding
        out = self.project(out)
        return out
