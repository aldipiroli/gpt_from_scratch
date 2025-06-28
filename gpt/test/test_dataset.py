import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile

from cfg import cfg
from dataset.tiny_shakespeare import Tinyshakespeare
from torch.utils.data import DataLoader
from utils.misc import get_logger

with tempfile.TemporaryDirectory() as tmp_dir:
    logger, _ = get_logger(tmp_dir)


def test_dataloader():
    dataset = Tinyshakespeare(cfg, logger, mode="train")
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )
    it = iter(data_loader)
    batch = next(it)
    context, target = batch
    assert (context[:, 1:] == target[:, :-1]).all()


if __name__ == "__main__":
    print("All test passed!")
