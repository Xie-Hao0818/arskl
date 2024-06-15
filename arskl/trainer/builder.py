from ..registry import TRAINER, build_from_cfg


def build_trainer(cfg, default_args=None):
    """Build a trainer from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    trainer = build_from_cfg(cfg, TRAINER, default_args)
    return trainer
