import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, gt)
        return loss
