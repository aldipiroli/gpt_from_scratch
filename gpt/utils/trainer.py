import torch
from torch.utils.data import DataLoader
from utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def reshuffle_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["DATA"]["batch_size"],
            shuffle=True,
        )

    def train(self):
        for epoch in range(self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
        self.generate_tokens()

    def train_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()
        train_loss = []
        for n_iter, (context, targets) in enumerate(self.train_loader):
            if n_iter > self.config["OPTIM"]["num_iterations_train"]:
                break
            self.optimizer.zero_grad()
            context = context.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(context)
            loss = self.loss_fn(preds, targets)
            train_loss.append(loss)

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
        self.logger.info(f"Train loss {torch.mean(torch.tensor(train_loss))}")

    @torch.no_grad()
    def evaluate_model(self, generate=False):
        self.model.eval()
        eval_loss = []
        for n_iter, (context, targets) in enumerate(self.val_loader):
            if n_iter > self.config["OPTIM"]["num_iterations_val"]:
                break
            context = context.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(context)
            loss = self.loss_fn(preds, targets)
            eval_loss.append(loss)
        self.logger.info(f"Val loss {torch.mean(torch.tensor(eval_loss))}")

    @torch.no_grad()
    def generate_tokens(self, num_gen_tokens=100, context="\n"):
        self.model.eval()
        context = self.val_dataset.tokenize_string(context)
        context = context.unsqueeze(0).to(self.device)
        generated = self.model.generate(context, num_gen_tokens)
        self.logger.info(f"Generated Tokens: {self.val_dataset.tokenizer.decode(generated.squeeze())}")
