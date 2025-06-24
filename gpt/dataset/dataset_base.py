import os
from abc import ABC
from pathlib import Path

import requests


class DatasetBase(ABC):
    def __init__(self, cfg, logger, mode="train"):
        self.cfg = cfg
        self.logger = logger
        self.root_dir = Path(cfg["root_dir"])
        os.makedirs(self.root_dir, exist_ok=True)
        self.target_url = cfg["target_url"]
        self.mode = mode
        self.download_data()
        self.data = self.load_data()

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
