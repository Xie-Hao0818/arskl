import torch.nn.functional as F
from lightning import LightningModule
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import accuracy

from arskl.model.builder import build_model
from .builder import LEARNER


@LEARNER.register_module()
class LearnerImg(LightningModule):
    def __init__(self, model_cfg, optim_cfg, hyper_cfg, epoch):
        super().__init__()
        self.lr = optim_cfg['lr']
        self.model = build_model(model_cfg)
        self.num_classes = model_cfg['num_classes']
        self.save_hyperparameters(hyper_cfg)
        self.optim_cfg = optim_cfg
        self.epoch = epoch

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        metrics = {"acc": acc, "loss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        metrics = {"vacc": acc, "vloss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optim = create_optimizer_v2(model_or_params=self.model, **self.optim_cfg)
        scheduler = CosineAnnealingLR(optimizer=optim, T_max=self.epoch)
        optim_dict = {'optimizer': optim, 'lr_scheduler': scheduler}
        return optim_dict
