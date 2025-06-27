import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger, now):
        super().__init__(config, logger, now)

    def reshuffle_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["DATA"]["batch_size"],
            shuffle=True,
        )

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
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
            self.write_float_to_tb(loss, "train/loss", self.total_iters)

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            pbar.set_postfix({"epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}", "loss": loss.item()})
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: train loss {torch.mean(torch.tensor(train_loss))}")

    @torch.no_grad()
    def evaluate_model(self):
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
        eval_loss = torch.tensor(eval_loss).mean()
        self.logger.info(f"Epoch {self.epoch}/{self.num_epochs}: val loss {torch.tensor(eval_loss)}")
        self.write_float_to_tb(eval_loss, "val/loss", self.epoch)

    def fine_tune(self):
        self.logger.info("Started fine-tuning..")
        for epoch in range(self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.fine_tune_one_epoch()
            self.evaluate_fine_tune_model()
            self.save_checkpoint()

    def fine_tune_one_epoch(self):
        self.model.train()
        self.reshuffle_dataloader()
        train_loss_sup = []
        train_loss_unsup = []
        max_iters = min(self.config["OPTIM"]["num_iterations_train"], len(self.train_loader))
        pbar = tqdm(enumerate(self.train_loader), total=max_iters)
        for n_iter, (context, target, question, answer, validity_mask) in pbar:
            if n_iter > max_iters:
                break
            self.optimizer.zero_grad()

            # unsupervised training
            context = context.to(self.device)
            target = target.to(self.device)
            preds = self.model(context)
            loss_uns = self.loss_fn(preds, target)
            train_loss_unsup.append(loss_uns)
            self.write_float_to_tb(loss_uns, "train/loss/loss_uns", self.total_iters)

            # supervised training
            question = question.to(self.device)
            answer = answer.to(self.device)
            validity_mask = validity_mask.to(self.device)
            preds = self.model(question)
            loss_sup = self.loss_fn(preds, answer, validity_mask)
            self.write_float_to_tb(loss_sup, "train/loss/loss_sup", self.total_iters)
            train_loss_sup.append(loss_sup)

            loss = 0.5 * loss_uns + loss_sup
            self.write_float_to_tb(loss, "train/loss/loss", self.total_iters)

            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            self.write_float_to_tb(self.get_lr(), "params/lr", self.total_iters)
            pbar.set_postfix({"loss_unsup": loss_uns.item(), "loss_sup": loss_sup.item(), "loss": loss.item()})
        self.logger.info(
            f"Epoch {self.epoch}/{self.num_epochs}: train loss_sup {torch.mean(torch.tensor(train_loss_sup))}, loss_unsup {torch.mean(torch.tensor(train_loss_unsup))}"
        )

    @torch.no_grad()
    def evaluate_fine_tune_model(self):
        self.model.eval()
        eval_loss_uns = []
        eval_loss_sup = []
        max_iters = min(self.config["OPTIM"]["num_iterations_val"], len(self.val_dataset))
        pbar = tqdm(enumerate(self.val_loader), total=max_iters)
        for n_iter, (context, target, question, answer, validity_mask) in pbar:
            if n_iter > self.config["OPTIM"]["num_iterations_val"]:
                break
            # unsupervised training
            context = context.to(self.device)
            target = target.to(self.device)
            preds = self.model(context)
            loss_uns = self.loss_fn(preds, target)
            eval_loss_uns.append(loss_uns)

            # supervised training
            question = question.to(self.device)
            answer = answer.to(self.device)
            validity_mask = validity_mask.to(self.device)
            preds = self.model(question)
            loss_sup = self.loss_fn(preds, answer, validity_mask)
            eval_loss_sup.append(loss_sup)

            pbar.set_postfix({"loss_unsup": loss_uns.item(), "loss_sup": loss_sup.item()})

        eval_loss_sup = torch.tensor(eval_loss_sup).mean()
        eval_loss_uns = torch.tensor(eval_loss_uns).mean()
        self.logger.info(
            f"Epoch {self.epoch}/{self.num_epochs}: val loss_sup {eval_loss_sup}, loss_unsup {eval_loss_uns}"
        )
        self.write_float_to_tb(eval_loss_sup, "val/loss/loss_sup", self.epoch)
        self.write_float_to_tb(eval_loss_uns, "val/loss/loss_unsup", self.epoch)
        self.write_float_to_tb(eval_loss_sup + eval_loss_uns, "val/loss/loss", self.epoch)

    @torch.no_grad()
    def generate_output(self, context="\n", num_gen_tokens=500):
        self.model.eval()
        self.logger.info("-" * 30)
        self.logger.info(f"Prompt: {context}")
        self.logger.info("-" * 30)
        self.logger.info("Generated Output")
        self.logger.info("-" * 30)
        context = self.val_dataset.tokenize_string(context[: self.model.context_len])
        context = context.unsqueeze(0).to(self.device)
        generated = self.model.generate(context, num_gen_tokens)
        generated_decode = self.val_dataset.tokenizer.decode(generated.squeeze())
        self.logger.info(f"{generated_decode}")
        self.write_text_to_tb(generated_decode, "generated/outputs", self.epoch)
        self.logger.info("-" * 30)
