import copy
from pathlib import Path
import runpy
from unittest.mock import patch

from mode_eci.io import read_jsonl, write_json
from mode_eci.pipeline import Experiment


def test_demo_stage_evaluation_needs_no_other_experts_or_gate(assets, tmp_path):
    config = copy.deepcopy(assets[0])
    config["output_dir"] = str(tmp_path)
    config["model"]["demo_pretraining"] = True
    experiment = Experiment(config)
    experiment.train_experts("final", ["D1", "D2", "D3", "D4"])
    evaluate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_demo_specialists.py"))["evaluate_specialists"]
    for split in ("dev", "test"):
        for shared in (True, False):
            directory, report = evaluate(experiment, split, shared)
            assert report["coverage"] == 1.0 and report["test_only"]
            predictions = list(read_jsonl(directory / "predictions.jsonl"))
            assert len(predictions) == report["overall"]["n"] == 12
            assert all(len(row["experts"]) == 1 for row in predictions)
            assert not any(b in row["experts"] for row in predictions for b in ("C", "S", "M"))
            if shared:
                assert all("D_all" in row["experts"] for row in predictions)
            else:
                assert all(row["features"]["demo_cell"] in row["experts"] for row in predictions)
            assert evaluate(experiment, split, shared)[1] == report
            if split == "test" and shared:
                old_report = copy.deepcopy(report)
                for group in ("overall", "intra", "cross"):
                    del old_report[group]["direction_fp_only"]
                write_json(directory / "metrics.json", old_report)
                with patch("mode_eci.experts.Expert.load", side_effect=AssertionError("Use saved predictions")):
                    assert evaluate(experiment, split, shared)[1] == report
    assert not (experiment.context("final")["directory"] / "experts/C").exists()
