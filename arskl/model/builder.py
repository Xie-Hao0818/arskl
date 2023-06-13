from mmcv.utils import Registry, build_from_cfg

MODEL = Registry('model')


def build_model(cfg, default_args=None):
    """Build a model from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    model = build_from_cfg(cfg, MODEL, default_args)
    return model
