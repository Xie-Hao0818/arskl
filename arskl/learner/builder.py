from ..registry import LEARNER, build_from_cfg


def build_learner(cfg, default_args=None):
    """Build a learner from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Learner: The constructed learner.
    """
    learner = build_from_cfg(cfg, LEARNER, default_args)
    return learner
