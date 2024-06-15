from timm.optim.lion import Lion
from ...registry import OPTIM

@OPTIM.register_module()
class Lion(Lion):
    def __init__(self, **kwargs):
        super(Lion, self).__init__(**kwargs)
