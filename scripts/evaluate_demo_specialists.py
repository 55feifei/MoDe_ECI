"""在其余专家和门控训练前，独立评估 D_all 及 D1–D4。"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from mode_eci.evaluation import evaluate
from mode_eci.experts import Expert
from mode_eci.io import claim_run, load_config, read_json, read_jsonl, write_json, write_jsonl
from mode_eci.pipeline import Experiment, release
from mode_eci.training import device_name, predict_expert


def evaluate_specialists(experiment, split, shared=False):
    documents = experiment.dev_docs if split == "dev" else experiment.test_docs
    context = experiment.context("final")
    branches = ["D_all"] if shared else ["D1", "D2", "D3", "D4"]
    mode = "D_all" if shared else "D_specialists"
    directory = experiment.root / "results" / f"{split}_{mode}_stage"
    checkpoints = {b: context["directory"] / "experts" / b / "checkpoint" for b in branches}
    claim_run(directory, {"split": split, "mode": mode,
                         "experiment": read_json(experiment.root / "manifest.json")["fingerprint"],
                         "checkpoints": {b: read_json(p / "expert.json")["checkpoint_fingerprint"]
                                         for b, p in checkpoints.items()}})
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    if metrics_path.is_file():
        report = read_json(metrics_path)
        if all("direction_fp_only" in report.get(group, {}) for group in ("overall", "intra", "cross")):
            return directory, report
    pairs = experiment.corpus.pairs_for(documents)
    if predictions_path.is_file():
        report = evaluate(pairs, list(read_jsonl(predictions_path)))
        write_json(metrics_path, report)
        return directory, report
    predictions = []
    features = experiment.get_features()
    for branch in branches:
        model = Expert.load(checkpoints[branch])
        builder = experiment.input_builder(model.tokenizer, model.head)
        rows = experiment.expert_rows(context, documents, branch, builder)
        model.to(device_name(experiment.config["training"]["device"]))
        output = predict_expert(model, rows, experiment.config["training"])
        for row, result in zip(rows, output):
            predictions.append({"pair_id": row["pair_id"], "doc_id": row["doc_id"],
                "features": features[row["pair_id"]], "test_only": experiment.config["test_only"],
                "experts": {branch: {**result, "coverage": row["coverage"], "retrieval": row["retrieval"]}},
                "probabilities": result["probabilities"], "label": int(np.argmax(result["probabilities"])),
                "branches": branches, "weights": [float(b == branch) for b in branches],
                "active_mask": [b == branch for b in branches], "status": "success"})
        release(model)
        del model
    report = evaluate(pairs, predictions)
    write_jsonl(predictions_path, predictions)
    write_json(metrics_path, report)
    print(f"EARLY_EVALUATION {split} {mode}: {report['overall']['direction']}", flush=True)
    return directory, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/roberta_mlm_dall_seed3407_gpu.yaml"))
    parser.add_argument("--fold", default="0")
    args = parser.parse_args()
    experiment = Experiment(load_config(args.config), args.fold)
    for split in ("dev", "test"):
        for shared in (True, False):
            evaluate_specialists(experiment, split, shared)


if __name__ == "__main__":
    main()
