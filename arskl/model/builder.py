from ..registry import MODEL, build_from_cfg


def build_model(cfg, default_args=None):
    """Build a model from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    model = build_from_cfg(cfg, MODEL, default_args)
    return model
