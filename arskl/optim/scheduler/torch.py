from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts,ReduceLROnPlateau,MultiStepLR

from ...registry import SCHEDULER


@SCHEDULER.register_module()
class CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, T_max, **kwargs):
        super(CosineAnnealingLR, self).__init__(T_max=T_max, **kwargs)


@SCHEDULER.register_module()
class CosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, **kwargs):
        super(CosineAnnealingWarmRestarts, self).__init__(**kwargs)


@SCHEDULER.register_module()
class OneCycleLR(OneCycleLR):
    def __init__(self, **kwargs):
        super(OneCycleLR, self).__init__(**kwargs)

@SCHEDULER.register_module()
class ReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super(ReduceLROnPlateau, self).__init__(**kwargs)

@SCHEDULER.register_module()
class MultiStepLR(MultiStepLR):
    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(**kwargs)