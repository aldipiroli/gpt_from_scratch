import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ffw = nn.Linear(10, 1)

    def forward(self, x):
        out = x
        return out
