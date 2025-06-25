import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred = pred.transpose(2, 1)  # (B, T, C) -> (B, C, T)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, gt)
        return loss
