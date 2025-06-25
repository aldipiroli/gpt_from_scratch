import os
from pathlib import Path

import numpy as np
import requests
import torch
from dataset.tokenizer import WordLevelTokenizer
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, cfg, logger, mode="train"):
        self.cfg = cfg
        self.cfg_data = cfg["DATA"]
        self.logger = logger
        self.root_dir = Path(self.cfg_data["root_dir"])
        os.makedirs(self.root_dir, exist_ok=True)
        self.target_url = self.cfg_data["target_url"]
        self.mode = mode
        self.download_data()

        self.data = self.load_data()
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)
        self.context_len = cfg["MODEL"]["context_len"]
        self.tokenizer = WordLevelTokenizer(self.vocab)

        self.data_train = self.data[: int(len(self.data) * 0.9)]
        self.data_val = self.data[int(len(self.data) * 0.9) :]
        self.data = self.data_train if mode == "train" else self.data_val

    def download_data(self):
        file_path = os.path.join(self.root_dir, "input.txt")
        if not os.path.exists(file_path):
            self.logger.info(f"Downloading: {self.target_url}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(requests.get(self.target_url).text)
            self.logger.info(f"Saved in: {file_path}")
        else:
            self.logger.info(f"File {file_path} already exists, skipping download.")

    def load_data(self):
        with open(os.path.join(self.root_dir, "input.txt"), "r") as f:
            data = f.read()
        return data

    def tokenize_string(self, x):
        x = np.array(list(x))
        x = self.tokenizer.encode(x)
        x = torch.tensor(x)
        return x

    def __len__(self):
        return len(self.data) - (self.context_len + 1)

    def __getitem__(self, idx):
        context = self.data[idx : idx + self.context_len]
        target = self.data[idx + 1 : idx + 1 + self.context_len]

        context = self.tokenize_string(context)
        target = self.tokenize_string(target)
        return context, target
