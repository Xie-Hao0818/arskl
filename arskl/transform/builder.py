from collections.abc import Sequence

from ..registry import TRANSFORM, build_from_cfg
from ..transform.compose import Compose


def build_transform(cfg, default_args=None):
    """Build a trainer from configs dict.

    Args:
        cfg (dict | list): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.
ssss
    Returns:
        Transform: The constructed transform.
    """
    if isinstance(cfg, Sequence):  # 配置为序列，需要组装，list为Sequence，dict不是Sequence
        transform = Compose(cfg)
    elif isinstance(cfg, dict):
        transform = build_from_cfg(cfg, TRANSFORM, default_args)
    return transform
