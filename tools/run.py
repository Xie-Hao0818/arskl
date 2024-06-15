import argparse
import os

import torch
from lightning.pytorch import seed_everything
import sys

sys.path.append('/root/autodl-tmp/arskl')
from Config import Config

from arskl.dataset.builder import build_dataset
from arskl.learner.builder import build_learner
from arskl.trainer.builder import build_trainer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 整理显存碎片
torch.cuda.empty_cache()  # 清空显存缓存
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
cfg = Config.fromfile(args.config)
seed_everything(seed=3407)  # 3407
if cfg.var['verbose']:
    print(cfg.pretty_text)
learner = build_learner(cfg.learner)
data = build_dataset(cfg.dataset)
trainer = build_trainer(cfg.trainer)
if cfg.test['is_test']:
    trainer.test(learner, data.test_dataloader(), cfg.test['ckpt_path'])
elif cfg.reload['is_reload']:
    trainer.fit(learner, data, ckpt_path=cfg.reload['ckpt_path'])
else:
    trainer.fit(learner, data)
