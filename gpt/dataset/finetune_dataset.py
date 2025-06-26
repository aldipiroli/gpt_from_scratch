import csv

import numpy as np
from dataset.tiny_shakespeare import Tinyshakespeare


class FineTuneDataset(Tinyshakespeare):
    def __init__(self, cfg, logger, mode):
        super().__init__(cfg, logger, mode)
        filepath = self.root_dir / self.cfg_data["finetune_data"]
        self.data = self.load_finetune_data(filepath)

        self.data_train = self.data[: int(len(self.data) * 0.9)]
        self.data_val = self.data[int(len(self.data) * 0.9) :]
        self.data = self.data_train if mode == "train" else self.data_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        text, validity_mask = self.truncate_text(data)

        context = text[: self.context_len]
        target = text[1 : self.context_len + 2]
        validity_mask = validity_mask[1 : 1 + self.context_len + 1]

        context = self.tokenize_string(context)
        target = self.tokenize_string(target)
        return context, target, validity_mask

    def truncate_text(self, data):
        question = data[0]
        answer = data[1]

        len_answer = len(answer)
        max_len_question = self.context_len - len_answer
        if max_len_question < 0:
            answer = answer[:-1]
            question = question[:1]
        else:
            question = question[:max_len_question]
        text = question + answer
        text = text.ljust(self.context_len + 1)
        validity_mask = np.zeros(len(text)).astype(np.float32)
        validity_mask[len(question) :] = 1
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


###########################################
import debugpy

debugpy.listen(("localhost", 6001))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
###########################################
