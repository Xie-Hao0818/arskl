import inspect
from typing import List

import timm.optim
import torch.optim

from ...registry import OPTIM, build_from_cfg


def register_torch_optimizers() -> List[str]:
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith(('__', 'Optimizer')):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIM.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()
'''
['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 
'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
'''


def register_timm_optimizers() -> List[str]:
    """Register optimizers in ``timm.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    timm_optimizers = []
    for module_name in dir(timm.optim):
        if module_name.startswith(('__', 'create', 'optim')) or module_name in TORCH_OPTIMIZERS:
            continue
        _optim = getattr(timm.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIM.register_module(module=_optim)
            timm_optimizers.append(module_name)
    return timm_optimizers


TIMM_OPTIMIZERS = register_timm_optimizers()
'''
['AdaBelief', 'Adafactor', 'Adahessian', 'AdamP', 'Adan', 'Lamb', 'Lars', 
'Lookahead', 'MADGRAD', 'Nadam', 'NvNovoGrad', 'RMSpropTF', 'SGDP']
'''


def build_optim(cfg, default_args=None):
    """Build a optim from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        optim: The constructed optim.
    """
    optim = build_from_cfg(cfg, OPTIM, default_args)
    return optim


'''
['Lion','SophiaG']
'''
