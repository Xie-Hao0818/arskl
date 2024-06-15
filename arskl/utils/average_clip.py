import torch.nn.functional as F


def average_clip(avg_type, cls_score):
    """Averaging class score over multiple clips.

    Using different averaging types ('score' or 'prob' or None, which defined in test_cfg) to computed the final
    averaged class score. Only called in test mode. By default, we use 'prob' mode.

    Args:
        avg_type (str): average_clip type
        cls_score (torch.Tensor): Class score to be averaged.

    Returns:
        torch.Tensor: Averaged class score.
    """
    assert len(cls_score.shape) == 3  # * (Batch, NumSegs, Dim)
    if avg_type not in ['score', 'prob', None]:
        raise ValueError(f'{avg_type} is not supported. Supported: ["score", "prob", None]')

    if avg_type is None:
        return cls_score

    if avg_type == 'prob':
        return F.softmax(cls_score, dim=2).mean(dim=1)
    elif avg_type == 'score':
        return cls_score.mean(dim=1)
