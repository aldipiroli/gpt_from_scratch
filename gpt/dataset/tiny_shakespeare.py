from dataset.dataset_base import DatasetBase


class Tinyshakespeare(DatasetBase):
    def __init__(self, cfg, logger, mode):
        super().__init__(cfg, logger, mode)
