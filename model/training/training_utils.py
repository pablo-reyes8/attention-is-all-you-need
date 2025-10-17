import time
import torch
PAD_ID = 0

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup = warmup_steps
        self._step_num = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self._step_num += 1
        scale = (self.d_model ** -0.5) * min(self._step_num ** -0.5,
                                             self._step_num * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


def token_acc(logits, tgt_out):
        preds = logits.argmax(dim=-1)
        mask = (tgt_out != PAD_ID)
        correct = (preds.eq(tgt_out) & mask).sum().item()
        total   = mask.sum().item()
        return (correct, total)