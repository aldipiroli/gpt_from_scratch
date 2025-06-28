import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, dropout):
        super().__init__()
        self.in_embed_size = in_embed_size
        self.out_embed_size = out_embed_size
        self.keys = nn.Linear(in_embed_size, out_embed_size, bias=False)
        self.query = nn.Linear(in_embed_size, out_embed_size, bias=False)
        self.values = nn.Linear(in_embed_size, out_embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)  # B,T,C

        qk = q @ k.transpose(2, 1) * self.out_embed_size**-0.5  # B, T, T

        tril = torch.tril(torch.ones(B, T, T)).to(qk.device)
        qk = qk.masked_fill(tril == 0, float("-inf"))
        attention = F.softmax(qk, -1)
        attention = self.dropout(attention)

        attention = attention @ v
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        assert embed_size % num_heads == 0
        self.heads = nn.ModuleList(
            [
                Attention(in_embed_size=embed_size, out_embed_size=embed_size // num_heads, dropout=dropout)
                for _ in range(num_heads)
            ]
        )
        self.project = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = out = torch.cat([h(x) for h in self.heads], -1)
        out = self.project(out)
        out = self.dropout(out)
        return out


class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.U = nn.Linear(embed_size, num_heads * 3 * (embed_size // num_heads), bias=False)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape
        C_h = self.embed_size // self.num_heads
        kqv = self.U(x)  # (B, T, n_heads*3*C_h)
        kqv = kqv.reshape(B, T, self.num_heads, C_h, 3)
        kqv = kqv.permute(2, 0, 1, 3, 4)  # (n_heads, B, T, C_h, 3)
        k = kqv[..., 0]
        q = kqv[..., 1]
        v = kqv[..., 2]  # (n_heads, B, T, C_h)

        qk = q @ k.transpose(3, 2) * C_h**-0.5  # (n_heads, B, T, T)
        tril = torch.tril(torch.ones(T, T)).to(qk.device)
        qk = qk.masked_fill(tril == 0, float("-inf"))
        attention = F.softmax(qk, -1)
        attention = self.dropout(attention)
        attention = attention @ v  # (n_heads, B, T, C_h)

        # aggregate heads
        attention = attention.permute(1, 2, 3, 0).reshape(B, T, C)

        out = self.project(attention)
        out = self.dropout(out)
        assert out.shape == (B, T, C)  # sanity check
        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        # self.mha = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout)
        self.mha_parallel = ParallelMultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffw = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4), nn.ReLU(), nn.Linear(4 * embed_size, embed_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        x_mha = self.mha_parallel(self.ln1(x))
        x = x + x_mha

        x_ffw = self.ffw(self.ln2(x))
        x = x + x_ffw
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_len = cfg["MODEL"]["context_len"]
        self.embed_size = cfg["MODEL"]["embed_size"]
        self.vocab_size = cfg["DATA"]["vocab_size"]
        self.transfomer_layers = cfg["MODEL"]["transfomer_layers"]
        self.num_heads = cfg["MODEL"]["num_heads"]
        self.dropout = cfg["MODEL"]["dropout"]

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embeddings = nn.Embedding(self.context_len, self.embed_size)
        self.tr_layers = nn.ModuleList(
            [
                TransformerLayer(self.embed_size, self.num_heads, dropout=self.dropout)
                for _ in range(self.transfomer_layers)
            ]
        )
        self.ln = nn.LayerNorm(self.embed_size)
        self.project = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.pos_embeddings(torch.arange(T, device=x.device))
        for tr_layer in self.tr_layers:
            x = tr_layer(x)
        x = self.ln(x)
        out = self.project(x)  # (B, T, vocab_size)
        return out

    @torch.no_grad()
    def generate(self, x, max_tokens):
        all_out_tokens = []
        for _ in range(max_tokens):
            out = self(x)  # (B, T)
            out_prob = F.softmax(out, -1)
            out_token = torch.multinomial(out_prob[:, -1, :], 1).to(x.device)
            all_out_tokens.append(out_token)
            x = torch.cat([x, out_token], -1)
            if x.shape[1] > self.context_len:
                x = x[:, -self.context_len :]  # remove extra tokens
        return torch.tensor(all_out_tokens)


if __name__ == "__main__":
    B, T, embed_size = 2, 10, 32
    num_heads = 4
    x = torch.randn(B, T, embed_size)
    module = GPTModel()
