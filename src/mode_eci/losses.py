"""专家和门控共用的等权 Focal Loss。"""


def focal_loss(log_probabilities, targets):
    """最后一维为类别，对每个目标计算 -(1 - p_t)^2 log(p_t) 后取均值。"""
    log_pt = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # expm1 在 p_t 接近 1 时也能准确计算 p_t - 1。
    return (-log_pt * log_pt.expm1().square()).mean()
