import copy
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from mode_eci.calibration import calibrate_records, decode_file
from mode_eci.cli import parser
from mode_eci.evaluation import bootstrap, evaluate, validation_metrics
from mode_eci.experts import Expert, collate
from mode_eci.features import RuleRouter
from mode_eci.gate import QueryGate, gate_loss, teacher_targets
from mode_eci.io import digest, load_config, read_json, read_jsonl, validate_config, write_json, write_jsonl
from mode_eci.pipeline import Experiment
from mode_eci.prompts import InputBuilder
from mode_eci.reporting import compare_files, summarize_files
from mode_eci.tasks import get_task


ROOT = Path(__file__).resolve().parents[1]
EXISTENCE = get_task("existence")


def binary_config(assets, directory):
    config = copy.deepcopy(assets[0])
    config.update(task="existence", output_dir=str(directory), experiment_id="test_only_existence")
    config["model"]["demo_pretraining"] = True
    return config


@pytest.mark.parametrize("head", ["bert_cls", "bert_mlm", "bart_relation", "bart_slots"])
def test_binary_heads_gradients_label_invariance_and_roundtrip(assets, tmp_path, head):
    config, corpus = assets
    settings = dict(config["model"], head=head)
    if head.startswith("bart"):
        settings["name"] = str(Path(settings["name"]).parent / "tiny_bart")
    model = Expert.create(settings, "existence").eval()
    builder = InputBuilder([model.tokenizer], head, config["input"], "existence")
    pair = corpus.pairs[0]
    query = corpus.query(pair)
    feature = RuleRouter("lexical")(query)
    demo = {"pair_id": pair["pair_id"], "query": query, "label": 1}
    row = builder.build(query, "D1", feature, [demo])
    reverse = builder.build(query, "D1", feature, [dict(demo, label=2)])
    assert row == reverse
    assert "[REL_12]" not in row["text"] and "[REL_21]" not in row["text"]
    batch = collate([row, row], model.tokenizer)
    labels = torch.tensor([0, EXISTENCE.label(1)])
    result = model(batch, labels)
    reverse_result = model(batch, torch.tensor([0, EXISTENCE.label(2)]))
    assert torch.equal(result["loss"], reverse_result["loss"])
    assert result["probabilities"].shape == (2, 2)
    assert torch.allclose(result["probabilities"].sum(-1), torch.ones(2))
    assert torch.isfinite(result["loss"])
    assert torch.allclose(result["probabilities"], model(batch, 1 - labels)["probabilities"])
    if head == "bart_slots":
        combinations = result["detail"]["combination_scores"].softmax(-1)
        assert torch.allclose(result["probabilities"][:, 1], combinations[:, 1:].sum(-1), atol=1e-6)
    result["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.save(tmp_path)
    assert read_json(tmp_path / "expert.json")["label_names"] == ["NONE", "CAUSAL"]
    loaded = Expert.load(tmp_path, "existence").eval()
    assert torch.allclose(result["probabilities"], loaded(batch)["probabilities"], atol=1e-6)
    with pytest.raises(ValueError, match="Task mismatch"):
        Expert.load(tmp_path, "direction")
    meta = read_json(tmp_path / "expert.json")
    meta["label_names"].reverse()
    write_json(tmp_path / "expert.json", meta)
    with pytest.raises(ValueError, match="label order"):
        Expert.load(tmp_path)


def test_old_checkpoints_default_to_direction_and_gate_checks_task(assets, tmp_path):
    config, _ = assets
    expert = Expert.create(config["model"])
    expert.save(tmp_path / "expert")
    gate = QueryGate.create(config["branches"], config["gate"])
    gate.save(tmp_path / "gate")
    for path in (tmp_path / "expert/expert.json", tmp_path / "gate/gate.json"):
        meta = read_json(path)
        meta.pop("task")
        meta.pop("label_names")
        write_json(path, meta)
    assert Expert.load(tmp_path / "expert").task.name == "direction"
    assert QueryGate.load(tmp_path / "gate").task.name == "direction"
    with pytest.raises(ValueError, match="Task mismatch"):
        QueryGate.load(tmp_path / "gate", "existence")


def test_config_defaults_guards_and_matching_roberta_settings(assets):
    config = copy.deepcopy(assets[0])
    validate_config(config)
    assert get_task(config.get("task", "direction")).num_labels == 3
    config["task"] = "bad"
    with pytest.raises(ValueError, match="Unknown task"):
        validate_config(config)
    config["task"] = "existence"
    config["graph"]["enabled"] = True
    with pytest.raises(ValueError, match="graph"):
        Experiment(config)
    config["graph"]["enabled"] = False
    config["model"].update(head="bart_slots", slot_relation_weight=0.5)
    with pytest.raises(ValueError, match="slot_relation_weight"):
        validate_config(config)
    config["model"]["slot_relation_weight"] = 0
    validate_config(config)
    original = yaml.safe_load((ROOT / "configs/roberta_mlm_dall_seed3407_gpu.yaml").read_text(encoding="utf-8"))
    binary = yaml.safe_load((ROOT / "configs/roberta_mlm_existence_dall_seed3407_gpu.yaml").read_text(encoding="utf-8"))
    assert binary.pop("task") == "existence"
    assert binary.pop("experiment_id") != original.pop("experiment_id")
    assert original == binary
    assert load_config(ROOT / "configs/roberta_mlm_existence_dall_seed3407_gpu.yaml")["task"] == "existence"


@pytest.mark.parametrize("base", ["bert_cls", "roberta_mlm", "bart_relation", "bart_slots"])
def test_generate_task_config_and_compare_cli(tmp_path, base):
    output = tmp_path / f"{base}.yaml"
    args = ["create_config.py", "--base", str(ROOT / "configs" / f"{base}.yaml"),
            "--output", str(output), "--task", "existence"]
    with patch.object(sys, "argv", args):
        runpy.run_path(str(ROOT / "scripts/create_config.py"), run_name="__main__")
    result = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert result["task"] == "existence"
    assert result["experiment_id"] != yaml.safe_load(Path(args[2]).read_text(encoding="utf-8"))["experiment_id"]
    assert parser().parse_args(["compare", "--first", "a", "--second", "b", "--output", "c", "--metric", "existence"]).metric == "existence"


def test_retrieval_is_identical_and_only_rendered_labels_change(assets, tmp_path):
    direction_config = copy.deepcopy(assets[0])
    direction_config["output_dir"] = str(tmp_path / "direction")
    original = Experiment(direction_config)
    binary = Experiment(binary_config(assets, tmp_path / "binary"))
    first_context, second_context = original.context("inner-0"), binary.context("inner-0")
    assert first_context["index"].metadata == second_context["index"].metadata
    first_plans = original.retrieval_plans(first_context, first_context["held"])
    second_plans = binary.retrieval_plans(second_context, second_context["held"])
    assert first_plans == second_plans
    model = Expert.create(binary.config["model"], "existence")
    builder = binary.input_builder(model.tokenizer, model.head)
    rows = binary.expert_rows(second_context, second_context["held"], "D_all", builder)
    assert {row["label"] for row in rows} <= {0, 1}
    for row in rows:
        raw = binary.corpus.by_id[row["pair_id"]]["label"]
        assert row["label"] == int(raw != 0)
        assert row["retrieval"] == first_plans[row["pair_id"]]
    shuffled = binary.expert_rows(second_context, second_context["held"], "D_all", builder, "shuffle_labels")
    for row in shuffled:
        assert row["retrieval"]["displayed_labels"] == [1 - int(d["label"] != 0) for d in row["retrieval"]["demos"]]
    positive = next(pair for pair in binary.corpus.pairs_for(second_context["held"]) if pair["label"] != 0)
    before = next(row for row in rows if row["pair_id"] == positive["pair_id"])
    positive["label"] = 3 - positive["label"]
    after = next(row for row in binary.expert_rows(second_context, second_context["held"], "D_all", builder)
                 if row["pair_id"] == positive["pair_id"])
    assert before == after


def prediction(identifier, label, probabilities, task="existence", distance=0):
    return {**get_task(task).metadata(), "pair_id": identifier, "doc_id": "doc",
            "label": label, "probabilities": probabilities,
            "features": {"sentence_distance": distance, "distance_bucket": distance, "demo_cell": "D4" if distance else "D2"},
            "experts": {"S": {"probabilities": probabilities, "coverage": {"truncated": False}, "retrieval": None}},
            "branches": ["S"], "weights": [1.0], "test_only": True, "status": "success"}


def test_binary_metrics_groups_diagnostics_and_empty_cases():
    pairs = [{"pair_id": str(i), "label": label} for i, label in enumerate([2, 1, 0, 0])]
    predictions = [prediction(str(i), label, q, distance=i % 2)
                   for i, (label, q) in enumerate([(1, [.2, .8]), (0, [.7, .3]), (1, [.1, .9]), (0, [.6, .4])])]
    report = evaluate(pairs, predictions)
    assert report["overall"]["existence"]["f1"] == 0.5
    assert "direction" not in report["overall"] and "direction_fp_only" not in report["cross"]
    assert report["label_counts"] == {"CAUSAL": 2, "NONE": 2}
    assert report["overall"]["nll"] == pytest.approx(-np.log([.8, .3, .1, .6]).mean())
    assert set(report["overall"]["class_reliability"]) == {"NONE", "CAUSAL"}
    assert report["D1"]["n"] == 0 and report["D1"]["existence"]["f1"] is None
    assert evaluate([], [], "existence")["overall"]["n"] == 0
    assert evaluate([{"pair_id": "0", "label": 0}], [prediction("0", 0, [.8, .2])])["overall"]["existence_average_precision"] is None
    predictions[0]["experts"]["S"]["retrieval"] = {"missing_labels": [1], "demos": [{"label": 2}]}
    diagnostics = evaluate(pairs, predictions)["input_diagnostics"]
    assert diagnostics["last_demo_follow_counts"]["last_demo_label_matches_expert"] == 1
    assert diagnostics["demo_missing_labels"] == {}  # 另一方向仍提供 CAUSAL 示例
    assert diagnostics["demo_missing_strata"] == {"E1_TO_E2": 1}
    value = validation_metrics([2, 0], [[.1, .9], [.8, .2]], "existence")
    assert value["selection_f1"] == value["existence_f1"] == 1
    assert "direction_f1" not in value
    with pytest.raises(ValueError, match="Task mismatch"):
        evaluate(pairs, predictions, "direction")
    bad = copy.deepcopy(predictions)
    bad[0]["probabilities"] = [.1, .2, .7]
    with pytest.raises(ValueError, match="dimension"):
        evaluate(pairs, bad)
    with pytest.raises(ValueError, match="missing"):
        evaluate(pairs, predictions[:-1])


@pytest.mark.parametrize("objective", ["G1", "G2", "G3"])
def test_binary_gate_teacher_and_all_wrong_rows(objective):
    q = torch.tensor([[[.2, .8], [.1, .9], [0., 0.]], [[.8, .2], [.7, .3], [0., 0.]]])
    labels = torch.tensor([EXISTENCE.label(2), EXISTENCE.label(1)])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    targets, covered = teacher_targets(q, labels, mask, 1.)
    assert covered.tolist() == [True, False]
    assert targets[1].sum() == 0
    assert torch.equal(targets, teacher_targets(q, torch.tensor([1, 1]), mask, 1.)[0])
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = gate_loss(logits, q, labels, mask, {"objective": objective, "temperature": 1., "teacher_temperature": 1., "teacher_weight": .5})
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.all(logits.grad[:, 2] == 0)
    if objective == "G2":
        with pytest.raises(ValueError, match="uncovered"):
            gate_loss(logits[1:], q[1:], labels[1:], mask[1:], {"objective": objective, "temperature": 1., "teacher_temperature": 1.})


def test_cross_task_comparison_projects_probabilities_and_keeps_decisions(assets, tmp_path):
    config, corpus = assets
    pair = next(pair for pair in corpus.pairs if pair["label"] == 2)
    direction = [prediction(pair["pair_id"], 0, [.4, .3, .3], "direction")]
    binary = [prediction(pair["pair_id"], 1, [.2, .8])]
    first, second = tmp_path / "direction.jsonl", tmp_path / "existence.jsonl"
    write_jsonl(first, direction)
    write_jsonl(second, binary)
    with pytest.raises(ValueError, match="Cross-task"):
        compare_files(config, first, second, tmp_path / "invalid.json")
    result = compare_files(config, first, second, tmp_path / "comparison.json", metric="existence")
    assert result["first"]["existence"]["f1"] == 0
    assert result["first"]["nll"] == pytest.approx(-np.log(.6))
    assert result["second"]["existence"]["f1"] == 1
    assert result["delta_existence_f1_interval_95"] == [1., 1.]
    assert list(read_jsonl(first)) == direction
    with pytest.raises(ValueError, match="Direction comparison"):
        bootstrap([pair], direction, binary, metric="direction")


@pytest.mark.parametrize("head", ["bert_mlm", "bart_slots"])
def test_binary_full_pipeline_pretraining_gates_calibration_and_reuse(assets, tmp_path, head):
    config = binary_config(assets, tmp_path / "artifacts")
    config["model"]["head"] = head
    if head.startswith("bart"):
        config["model"]["name"] = str(Path(config["model"]["name"]).parent / "tiny_bart")
    experiment = Experiment(config)
    experiment.run()
    before = {str(path): path.stat().st_mtime_ns for path in experiment.root.rglob("training.json")}
    experiment.run()
    assert before == {str(path): path.stat().st_mtime_ns for path in experiment.root.rglob("training.json")}
    for variant in ("no_demo", "shuffle_labels"):
        _, diagnostic = experiment.predict("dev", "equal", demo_variant=variant)
        assert diagnostic["coverage"] == 1
        if variant == "no_demo":
            assert diagnostic["input_diagnostics"]["last_demo_follow_counts"].get("n", 0) == 0
    supervised = experiment.supervised_oof()
    assert {row["label"] for row in supervised} == {0, 1}
    for objective in ("G1", "G2", "G3"):
        gate_dir = experiment.gate_directory(objective)
        gate = QueryGate.load(gate_dir / "checkpoint", "existence")
        assert gate.task == EXISTENCE
        coverage = read_json(gate_dir / "coverage.json")
        assert set(coverage["by_gold"]) == {"NONE", "CAUSAL"}
        history = read_json(gate_dir / "selection/history.json")
        assert "existence_f1" in history[0]["validation"]
        best_epoch = read_json(gate_dir / "selection/training.json")["best_epoch"]
        split = read_json(gate_dir / "validation_split.json")
        assert set(split["training_documents"]).isdisjoint(split["validation_documents"])
        validation_docs = set(split["validation_documents"])
        validation = experiment.gate_rows([r for r in supervised if r["doc_id"] in validation_docs],
                                          InputBuilder([gate.tokenizer], "bert_cls", config["input"], EXISTENCE))
        probabilities, _ = experiment.gate_probabilities(gate, validation)
        scores = validation_metrics([r["label"] for r in validation], probabilities, EXISTENCE)
        assert scores == pytest.approx(history[best_epoch - 1]["validation"])
        directory, report = experiment.predict("test", "learned", objective)
        assert report["coverage"] == 1 and report["test_only"] and report["task"] == "existence"
        predictions = list(read_jsonl(directory / "predictions.jsonl"))
        assert all(len(row["probabilities"]) == 2 for row in predictions)
    for scope in ("inner-0", "inner-1", "final"):
        context = experiment.context(scope)
        parent = context["directory"] / "experts/D_all/checkpoint"
        parent_fingerprint = read_json(parent / "expert.json")["checkpoint_fingerprint"]
        for branch in ("D1", "D2", "D3", "D4"):
            provenance = read_json(context["directory"] / "experts" / branch / "provenance.json")
            assert provenance["initialization_fingerprint"] == parent_fingerprint
        assert not set(context["train"]) & set(context["held"])
    summary = summarize_files(config, [directory / "predictions.jsonl"], tmp_path / "summary.json")
    assert summary["pooled_existence_f1_mean"] == report["overall"]["existence"]["f1"]
    assert "pooled_direction_f1_mean" not in summary
    wrong = copy.deepcopy(config)
    wrong["task"] = "direction"
    with pytest.raises(ValueError, match="Task mismatch"):
        summarize_files(wrong, [directory / "predictions.jsonl"], tmp_path / "mixed.json")
    decoded = decode_file(config, directory / "predictions.jsonl", tmp_path / "decoded", "existence", 0.)
    assert decoded["overall"]["existence"]["fn"] == 0
    with pytest.raises(ValueError, match="Paper direction"):
        decode_file(config, directory / "predictions.jsonl", tmp_path / "paper", "paper", .5, "S")
    experiment.expert_predictions("inner-0", experiment.test_docs, "calibration_test")
    scope_dir = experiment.root / "inner-0/predictions"
    calibrated = calibrate_records(config, scope_dir / "held/records.jsonl", scope_dir / "calibration_test/records.jsonl", tmp_path / "calibrated")
    assert calibrated["task"] == "existence" and set(calibrated["fits"]) == set(config["branches"])
    assert all(row["nll_after"] <= row["nll_before"] + 1e-8 for row in calibrated["fits"].values())
    evaluate_specialists = runpy.run_path(str(ROOT / "scripts/evaluate_demo_specialists.py"))["evaluate_specialists"]
    for shared in (True, False):
        location, stage = evaluate_specialists(experiment, "test", shared)
        assert stage["task"] == "existence" and stage["coverage"] == 1
        with patch("mode_eci.experts.Expert.load", side_effect=AssertionError("Reuse predictions")):
            assert evaluate_specialists(experiment, "test", shared)[1] == stage
    # OOF 的任务标签即使在重新计算校验和后也不能混用。
    records_path = experiment.root / "oof/records.jsonl"
    records = list(read_jsonl(records_path))
    records[0]["task"] = "direction"
    write_jsonl(records_path, records)
    provenance_path = experiment.root / "oof/provenance.json"
    provenance = read_json(provenance_path)
    provenance["records_fingerprint"] = digest(records)
    write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="Task mismatch"):
        experiment.supervised_oof()
