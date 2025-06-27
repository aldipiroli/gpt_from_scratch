import csv
import random

import numpy as np
from dataset.tiny_shakespeare import Tinyshakespeare


class FineTuneDataset(Tinyshakespeare):
    def __init__(self, cfg, logger, mode):
        super().__init__(cfg, logger, mode)
        filepath = self.root_dir / self.cfg_data["finetune_data"]
        self.data_qa = self.load_finetune_data(filepath)

        self.data_qa_train = self.data_qa[: int(len(self.data_qa) * 0.9)]
        self.data_qa_val = self.data_qa[int(len(self.data_qa) * 0.9) :]
        self.logger.info(f"Training data size: {len(self.data_qa_train)}")
        self.logger.info(f"Validation data size: {len(self.data_qa_val)}")
        self.data_qa = self.data_qa_train if mode == "train" else self.data_qa_val

    def __getitem__(self, idx):
        # unsupervised
        context = self.data[idx : idx + self.context_len]
        target = self.data[idx + 1 : idx + 1 + self.context_len]
        context = self.tokenize_string(context)
        target = self.tokenize_string(target)

        # supervised
        idx_sup = random.randint(0, len(self.data_qa) - 1)
        data = self.data_qa[idx_sup]
        question = data[0]
        answer = data[1]
        question, _ = self.truncate_text(question)
        answer, validity_mask = self.truncate_text(answer)
        question = self.tokenize_string(question)
        answer = self.tokenize_string(answer)

        return context, target, question, answer, validity_mask

    def truncate_text(self, text):
        if len(text) > self.context_len:
            text = text[: self.context_len]

        len_text = len(text)
        text = text.ljust(self.context_len)
        validity_mask = np.zeros(len(text)).astype(np.float32)
        validity_mask[:len_text] = 1
        return text, validity_mask

    def load_finetune_data(self, filepath):
        qa_list = []
        with open(filepath, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header line
            for row in reader:
                question, answer = row
                qa_list.append((question, answer))
        return qa_list
