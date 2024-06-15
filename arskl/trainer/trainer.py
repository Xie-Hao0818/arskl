from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateFinder, RichModelSummary, LearningRateMonitor, RichProgressBar, \
    ModelCheckpoint

from ..registry import TRAINER
from lightning.pytorch.profilers import PyTorchProfiler

# profiler = PyTorchProfiler(filename="perf_logs")
profiler = None


@TRAINER.register_module()
class Trainer(Trainer):
    def __init__(self, ckpt_cfg, **kwargs):
        super().__init__(callbacks=[
            RichModelSummary(),
            LearningRateMonitor(),
            RichProgressBar(),
            ModelCheckpoint(**ckpt_cfg),
        ], **kwargs, profiler=profiler)
