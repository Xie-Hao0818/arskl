import torch
import torch.nn as nn
from lightning import LightningModule
from pycm import ConfusionMatrix
from torchmetrics.functional import accuracy

from ..loss.distillation_loss import FocalLoss
from ..model import build_model
from ..optim.optim import build_optim
from ..optim.scheduler import build_scheduler
from ..registry import LEARNER


@LEARNER.register_module()
class Learner_Skl(LightningModule):

    def __init__(self, model_cfg, optim_cfg, scheduler_cfg, hyper_cfg, teacher_cfg):
        super().__init__()
        self.lr = optim_cfg['lr']
        self.model = build_model(model_cfg)
        if hyper_cfg['compile']:
            self.model = torch.compile(self.model)
        self.model.init_weights()
        self.num_classes = hyper_cfg['num_classes']
        self.save_hyperparameters(hyper_cfg)
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=hyper_cfg['label_smoothing'])
        self.base_loss1 = FocalLoss()
        # self.loss = DiscriminationLoss4(self.base_loss, hyper_cfg['distillation_type'], alpha=10, tau=10,
        #                                 teacher_cfg=teacher_cfg, num_classes=self.num_classes)
        self.y_true = []
        self.y_pred = []

    def training_step(self, batch, batch_idx):
        x, y = batch['imgs'], batch['label']
        y = torch.squeeze(y)
        y_hat = self.model(x)
        # loss = self.loss(x, y_hat, y)
        loss = self.base_loss1(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        metrics = {"acc": acc, "loss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['imgs'], batch['label']
        y = torch.squeeze(y)
        y_hat = self.model(x)
        # loss = self.loss(x, y_hat, y)
        # loss = self.base_loss(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        # metrics = {"vacc": acc, "vloss": loss}
        metrics = {"vacc": acc, }
        self.log_dict(metrics, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch['imgs'], batch['label']
        y = torch.squeeze(y)
        y_hat = self.model(x)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        # self.log(name='acc', value=acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(name='acc', value=acc, prog_bar=True)
        self.y_true = torch.cat([torch.tensor(self.y_true).cuda(), y], 0)
        self.y_pred = torch.cat([torch.tensor(self.y_pred).cuda(), torch.argmax(y_hat, dim=1)], 0)

    def on_test_end(self):
        cm = ConfusionMatrix(actual_vector=self.y_true.cpu().numpy(), predict_vector=self.y_pred.cpu().numpy())
        cm.save_csv(name='/root/autodl-tmp/matrix/ntu60_931')

    def configure_optimizers(self):
        self.optim_cfg['params'] = self.parameters()
        optim = build_optim(self.optim_cfg)
        self.scheduler_cfg['optimizer'] = optim
        scheduler = build_scheduler(self.scheduler_cfg)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "vacc",
            "strict": True,
            "name": None,
        }
        optim_dict = {'optimizer': optim, 'lr_scheduler': lr_scheduler_config}
        return optim_dict
