from ...registry import SCHEDULER, build_from_cfg


def build_scheduler(cfg, default_args=None):
    """Build a scheduler from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        scheduler: The constructed scheduler.
    """
    scheduler = build_from_cfg(cfg, SCHEDULER, default_args)
    return scheduler
