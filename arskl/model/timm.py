import timm
import torch.nn as nn
from arskl.model.builder import MODEL


@MODEL.register_module()
class TimmModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model(**kwargs)

    def forward(self, x):
        return self.model(x)
