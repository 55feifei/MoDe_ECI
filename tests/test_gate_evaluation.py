import copy

import numpy as np
import pytest
import torch

from mode_eci.evaluation import align_predictions, prf
from mode_eci.gate import gate_loss, masked_weights, teacher_targets
from mode_eci.io import claim_run


def test_teacher_values_mask_and_all_wrong():
    probabilities = torch.tensor([[[.10, .80, .10], [.15, .70, .15], [.55, .35, .10], [.40, .25, .35]],
                                  [[.50, .40, .10], [.10, .40, .50], [0., 0., 0.], [0., 0., 0.]]])
    labels = torch.tensor([1, 1])
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    targets, covered = teacher_targets(probabilities, labels, mask, 1.0)
    assert np.allclose(targets[0], [.524979, .475021, 0, 0], atol=1e-6)
    assert covered.tolist() == [True, False]
    assert targets[1].sum() == 0
    logits = torch.zeros(2, 4, requires_grad=True)
    settings = {"temperature": 1., "teacher_temperature": 1., "teacher_weight": .5, "objective": "G3"}
    loss = gate_loss(logits, probabilities, labels, mask, settings)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[1, 2:] == 0)
    # 全错专家也能经融合得到正确标签，故覆盖率不是融合理论上界。
    assert probabilities[1, :2].mean(0).argmax() == 1


@pytest.mark.parametrize("objective", ["G1", "G2", "G3"])
def test_gate_losses_are_finite(objective):
    logits = torch.tensor([[1., 0., 3.]], requires_grad=True)
    q = torch.tensor([[[.7, .2, .1], [.1, .8, .1], [0., 0., 0.]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = gate_loss(logits, q, torch.tensor([1]), mask,
                     {"objective": objective, "temperature": 1., "teacher_temperature": .1, "teacher_weight": .5})
    loss.backward()
    assert q.grad is None
    assert torch.isfinite(logits.grad).all()
    weights = masked_weights(logits, mask, top_k=1)
    assert weights.tolist() == [[1., 0., 0.]]


def test_wrong_direction_counts_fp_and_fn_and_coverage_is_strict():
    value = prf([1, 0], [2, 0], directional=True)
    assert (value["tp"], value["fp"], value["fn"]) == (0, 1, 1)
    assert prf([1], [2])["f1"] == 1
    assert prf([0], [0], True)["f1"] is None
    with pytest.raises(ValueError, match="missing"):
        align_predictions([{"pair_id": "q"}], [])


def test_experiment_config_reuse_is_rejected(tmp_path):
    claim_run(tmp_path, {"config": "a"})
    claim_run(tmp_path, {"config": "a"})
    with pytest.raises(ValueError, match="identity"):
        claim_run(tmp_path, {"config": "b"})
