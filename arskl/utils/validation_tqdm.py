import sys

import lightning
from tqdm import tqdm


class Bar(lightning.pytorch.callbacks.TQDMProgressBar):
    def __init__(self):
        super().__init__()

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
