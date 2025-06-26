import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    def fine_tune(self):
        self.logger.info("Started fine-tuning..")
        for epoch in range(self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.fine_tune_one_epoch()
            self.save_checkpoint()

    def fine_tune_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()
        train_loss = []
        max_iters = min(self.config["OPTIM"]["num_iterations_train"], len(self.train_loader))
        pbar = tqdm(enumerate(self.train_loader), total=max_iters)
        for n_iter, (context, targets, validity_mask) in pbar:
            if n_iter > max_iters:
                break
            self.optimizer.zero_grad()
            context = context.to(self.device)
            targets = targets.to(self.device)
            validity_mask = validity_mask.to(self.device)

            preds = self.model(context)
            loss = self.loss_fn(preds, targets, validity_mask)
            train_loss.append(loss)

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            pbar.set_postfix({"loss": loss.item()})
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: train loss {torch.mean(torch.tensor(train_loss))}")

    def train(self):
        self.logger.info("Started training..")
        for epoch in range(self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            self.save_checkpoint()
            if epoch % self.eval_every == 0:
                self.generate_output()

    def train_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()
        train_loss = []
        max_iters = min(self.config["OPTIM"]["num_iterations_train"], len(self.train_loader))
        pbar = tqdm(enumerate(self.train_loader), total=max_iters)
        for n_iter, (context, targets) in pbar:
            if n_iter > max_iters:
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
            pbar.set_postfix({"loss": loss.item()})
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: train loss {torch.mean(torch.tensor(train_loss))}")

    @torch.no_grad()
    def evaluate_model(self, generate=False):
        self.model.eval()
        eval_loss = []
        max_iters = min(self.config["OPTIM"]["num_iterations_val"], len(self.val_dataset))
        pbar = tqdm(enumerate(self.val_loader), total=max_iters)
        for n_iter, (context, targets) in pbar:
            if n_iter > self.config["OPTIM"]["num_iterations_val"]:
                break
            context = context.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(context)
            loss = self.loss_fn(preds, targets)
            eval_loss.append(loss)
            pbar.set_postfix({"loss": loss.item()})
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: val loss {torch.mean(torch.tensor(eval_loss))}")

    @torch.no_grad()
    def generate_output(self, context="\n", num_gen_tokens=2000):
        self.model.eval()
        self.logger.info("-" * 30)
        self.logger.info(f"Prompt: {context}")
        self.logger.info("-" * 30)
        self.logger.info("Generated Output")
        self.logger.info("-" * 30)
        context = self.val_dataset.tokenize_string(context[: self.model.context_len])
        context = context.unsqueeze(0).to(self.device)
        generated = self.model.generate(context, num_gen_tokens)
        self.logger.info(f"{self.val_dataset.tokenizer.decode(generated.squeeze())}")
        self.logger.info("-" * 30)
