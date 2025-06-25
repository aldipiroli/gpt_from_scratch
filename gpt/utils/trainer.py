from tqdm import tqdm
from utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

    def train(self):
        epoch_start = self.epoch
        for curr_epoch in range(epoch_start, self.optim_config["num_epochs"]):
            self.train_one_epoch()
            if (curr_epoch + 1) % self.eval_every == 0:
                self.evaluate_model()
                self.save_checkpoint()
            self.epoch = curr_epoch

    def train_one_epoch(self):
        self.model.train()
        with tqdm(enumerate(self.train_loader), desc=f"Epoch: {self.epoch}/{self.optim_config['num_epochs']}") as pbar:
            for n_iter, (context, targets) in pbar:
                self.optimizer.zero_grad()
                context = context.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(context)
                loss = self.loss_fn(preds, targets)

                loss.backward()
                self.gradient_clip()
                self.optimizer.step()
                self.total_iters += 1
                pbar.set_postfix({"total_iters": self.total_iters, "loss": loss.item(), "lr": self.get_lr()})
                self.scheaduler_step()

    def evaluate_model(self):
        self.logger.info("Running Evaluation...")
        self.model.eval()
        for n_iter, (tokens, targets) in enumerate(self.val_loader):
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(tokens)
            loss = self.loss_fn(preds, targets)
