import torch.nn.functional as F
from lightning import LightningModule
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from arskl.model.builder import build_model
from .builder import LEARNER


@LEARNER.register_module()
class LearnerImg(LightningModule):
    def __init__(self, model_cfg, optim_cfg, hyper_cfg, epoch):
        super().__init__()
        self.lr = optim_cfg['lr']
        self.model = build_model(model_cfg)
        self.swa_model = AveragedModel(self.model)
        self.optim = create_optimizer_v2(model_or_params=self.model, **optim_cfg)
        self.scheduler = CosineAnnealingLR(optimizer=self.optim, T_max=epoch)
        self.num_classes = model_cfg['num_classes']
        self.save_hyperparameters(hyper_cfg)

    def on_before_optimizer_step(self, optimizer):
        optimizer.param_groups[0]['lr'] = self.lr  # 将找到的最佳学习率重新赋值

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        lr = self.scheduler.get_last_lr()[0]  # 获取最新学习率
        metrics = {"acc": acc, "loss": loss, 'lr': lr}
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
        optim_dict = {'optimizer': self.optim, 'lr_scheduler': self.scheduler}
        return optim_dict

    def on_train_epoch_end(self):
        self.swa_model.update_parameters(self.model)

    def on_train_end(self):
        update_bn(self.trainer.datamodule.train_dataloader(), self.swa_model, device=self.device)
