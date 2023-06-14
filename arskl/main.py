from mmcv import Config

from arskl.data.builder import build_dataset
from arskl.learner.builder import build_learner
from arskl.trainer.builder import build_trainer
import torch

torch.set_float32_matmul_precision('medium')
cfg = Config.fromfile('config/resnet18d_cifar10_timm.py')
print(cfg.pretty_text)
learner = build_learner(cfg.learner)
data = build_dataset(cfg.dateset)
trainer = build_trainer(cfg.trainer)
trainer.fit(learner, data)
