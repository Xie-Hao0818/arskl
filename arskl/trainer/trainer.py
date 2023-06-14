from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateFinder

from arskl.trainer.builder import TRAINER
from arskl.utils.validation_tqdm import Bar


@TRAINER.register_module()
class Trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(callbacks=[Bar(), LearningRateFinder()], **kwargs)
