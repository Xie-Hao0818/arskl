from mmcv.utils import Registry, build_from_cfg

TRANSFORM = Registry('transform')


def build_transform(cfg, default_args=None):
    """Build a trainer from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.
ssss
    Returns:
        Transform: The constructed transform.
    """
    transform = build_from_cfg(cfg, TRANSFORM, default_args)
    return transform
