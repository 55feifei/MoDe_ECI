import copy
from pathlib import Path
import shutil

import pytest
import torch

import mode_eci.pipeline as pipeline
from mode_eci.experts import Expert
from mode_eci.features import RuleRouter
from mode_eci.io import read_json, validate_config, write_jsonl
from mode_eci.pipeline import Experiment
from mode_eci.reporting import summarize_files


def demo_experiment(assets, tmp_path):
    config = copy.deepcopy(assets[0])
    config["output_dir"] = str(tmp_path)
    config["model"]["demo_pretraining"] = True
    return Experiment(config)


def test_all_demo_rows_match_specialist_union_and_exclude_held_documents(assets, tmp_path):
    experiment = demo_experiment(assets, tmp_path)
    context = experiment.context("inner-0")
    model = Expert.create(experiment.config["model"])
    builder = experiment.input_builder(model.tokenizer, model.head)
    all_rows = experiment.expert_rows(context, context["train"], "D_all", builder)
    split_rows = [row for branch in ("D1", "D2", "D3", "D4")
                  for row in experiment.expert_rows(context, context["train"], branch, builder)]
    assert {r["pair_id"]: r for r in all_rows} == {r["pair_id"]: r for r in split_rows}
    assert len(all_rows) == len(experiment.corpus.pairs_for(context["train"]))
    assert {r["demo_cell"] for r in all_rows} == {"D1", "D2", "D3", "D4"}
    for row in all_rows:
        assert row["doc_id"] not in context["held"]
        for demo in row["retrieval"]["demos"]:
            record = context["index"].by_id[demo["pair_id"]]
            assert record["doc_id"] in context["train"]
            assert record["doc_id"] != row["doc_id"]
            assert record["features"]["demo_cell"] == row["demo_cell"]


def test_pretraining_initializes_specialists_and_full_oof_pipeline_reuses_stages(assets, tmp_path, monkeypatch):
    experiment = demo_experiment(assets, tmp_path)
    original_fit = pipeline.fit
    initial_weights = {}

    def record_fit(model, rows, tokenizer, settings, seed, directory, *args, **kwargs):
        if isinstance(model, Expert):
            initial_weights[str(directory)] = model.backbone.get_input_embeddings().weight.detach().cpu().clone()
        return original_fit(model, rows, tokenizer, settings, seed, directory, *args, **kwargs)

    monkeypatch.setattr(pipeline, "fit", record_fit)
    experiment.run(objectives=("G3",))
    assert experiment.plan()["expert_training_jobs"] == 24  # 3 scopes × (7 experts + D_all)
    for scope in ("inner-0", "inner-1", "final"):
        context = experiment.context(scope)
        experts = context["directory"] / "experts"
        parent = Expert.load(experts / "D_all" / "checkpoint")
        provenance = read_json(experts / "D_all" / "provenance.json")
        assert set(provenance["train_documents"]) == set(context["train"])
        assert not set(provenance["train_documents"]) & set(context["held"])
        assert len(provenance["training_pair_ids"]) == sum(provenance["training_cells"].values())
        for branch in ("D1", "D2", "D3", "D4"):
            child = read_json(experts / branch / "provenance.json")
            assert Path(child["initialization"]) == experts / "D_all" / "checkpoint"
            assert child["initialization_fingerprint"] == read_json(experts / "D_all" / "checkpoint" / "expert.json")["checkpoint_fingerprint"]
            assert torch.equal(initial_weights[str(experts / branch)], parent.backbone.get_input_embeddings().weight)
            assert child["training_cells"][branch] == len(child["training_pair_ids"])
    before = {str(p): p.stat().st_mtime_ns for p in experiment.root.rglob("training.json")}
    experiment.run(objectives=("G3",))
    assert before == {str(p): p.stat().st_mtime_ns for p in experiment.root.rglob("training.json")}
    directory, report = experiment.predict("test", "learned", "G3")
    assert report["test_only"] and report["coverage"] == 1.0
    assert "D_all" not in experiment.config["branches"]
    summary = summarize_files(config=experiment.config, paths=[directory / "predictions.jsonl"],
                              output=tmp_path / "summary.json")
    seed_report = summary["seeds"][str(experiment.config["seed"])]
    fold_report = seed_report["folds"][experiment.fold]
    assert fold_report["intra"] == report["intra"] and fold_report["cross"] == report["cross"]
    for group in ("overall", "intra", "cross"):
        for metric in ("direction", "direction_fp_only"):
            score = report[group][metric]["f1"]
            assert summary["pooled_f1"][group][metric] == {"mean": score, "std": None}


def test_demo_pretraining_config_requires_demo_branches(assets):
    config = copy.deepcopy(assets[0])
    config["model"]["demo_pretraining"] = True
    config["branches"] = ["S"]
    with pytest.raises(ValueError, match="requires at least one D branch"):
        validate_config(config)
    config["branches"] = ["D1"]
    validate_config(config)


def test_empty_specialist_retains_parent_and_records_no_finetuning(assets, tmp_path):
    config, corpus = assets
    config = copy.deepcopy(config)
    destination = tmp_path / "data"
    shutil.copytree(config["data_dir"], destination)
    router = RuleRouter("lexical")
    write_jsonl(destination / "pairs.jsonl", [p for p in corpus.pairs
                if router(corpus.query(p))["demo_cell"] != "D3"])
    config.update(data_dir=str(destination), output_dir=str(tmp_path / "artifacts"), branches=["D3"])
    config["model"]["demo_pretraining"] = True
    experiment = Experiment(config)
    experiment.train_experts()
    experts = experiment.context("final")["directory"] / "experts"
    summary = read_json(experts / "D3" / "training.json")
    assert summary["training_mode"] == "initialized_only"
    assert summary["epochs_run"] == summary["training_samples"] == 0
    parent = Expert.load(experts / "D_all" / "checkpoint")
    child = Expert.load(experts / "D3" / "checkpoint")
    assert all(torch.equal(value, child.state_dict()[key]) for key, value in parent.state_dict().items())
