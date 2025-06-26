import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        pred = pred.transpose(2, 1)  # (B, T, C) -> (B, C, T)
        B, C, T = pred.shape
        criterion = nn.CrossEntropyLoss(reduction="none")
        if mask is not None:
            loss = criterion(pred, gt)
            loss = loss * mask
        else:
            loss = criterion(pred, gt)
        loss = loss.mean()
        return loss
