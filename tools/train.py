import argparse

import torch
from lightning.pytorch import seed_everything
from mmcv import Config

from arskl.dataset.builder import build_dataset
from arskl.learner.builder import build_learner
from arskl.trainer.builder import build_trainer

torch.set_float32_matmul_precision('medium')
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
print(cfg.pretty_text)
seed_everything(cfg.var['seed'])
learner = build_learner(cfg.learner)
data = build_dataset(cfg.dateset)
trainer = build_trainer(cfg.trainer)
trainer.fit(learner, data)
