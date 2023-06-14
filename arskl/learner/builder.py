from mmcv.utils import Registry, build_from_cfg

LEARNER = Registry('learner')


def build_learner(cfg, default_args=None):
    """Build a learner from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    learner = build_from_cfg(cfg, LEARNER, default_args)
    return learner
