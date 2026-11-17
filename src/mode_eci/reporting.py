"""跨折跨种子的汇总及配对比较，不把开发预测混入正式主表。"""

from pathlib import Path

import numpy as np

from .data import Corpus
from .evaluation import bootstrap, evaluate
from .io import read_json, read_jsonl, write_json, write_jsonl


def compare_files(config, first, second, output, by_topic=False):
    corpus = Corpus(config["data_dir"])
    first_rows, second_rows = list(read_jsonl(first)), list(read_jsonl(second))
    ids = {r["pair_id"] for r in first_rows}
    pairs = [p for p in corpus.pairs if p["pair_id"] in ids]
    result = bootstrap(pairs, first_rows, second_rows, seed=config["seed"], by_topic=by_topic)
    result["first"] = evaluate(pairs, first_rows)["overall"]
    result["second"] = evaluate(pairs, second_rows)["overall"]
    result["test_only"] = any(r["test_only"] for r in first_rows + second_rows)
    write_json(output, result)
    return result


def summarize_files(config, paths, output):
    corpus = Corpus(config["data_dir"])
    expected = {p["pair_id"] for fold in range(len(corpus.splits["folds"])) for p in corpus.split(fold, "test")}
    by_seed, reports = {}, {}
    definition = None
    for path in paths:
        manifest = read_json(Path(path).parent / "manifest.json")
        if manifest["split"] != "test":
            raise ValueError("Formal pooled summary requires test predictions")
        current = {k: manifest[k] for k in ("experiment_id", "head", "fusion", "objective", "demo_variant", "top_k", "drop", "gate_settings")}
        if definition is not None and current != definition:
            raise ValueError("Cannot pool different experimental systems")
        definition = current
        rows = list(read_jsonl(path))
        seed = str(manifest["seed"])
        by_seed.setdefault(seed, []).extend(rows)
        ids = {r["pair_id"] for r in rows}
        pairs = [p for p in corpus.pairs if p["pair_id"] in ids]
        report = evaluate(pairs, rows)
        reports.setdefault(seed, {})[manifest["fold"]] = {
            **report["overall"], "intra": report["intra"], "cross": report["cross"]}
    result = {"definition": definition, "seeds": {}}
    pairs = [p for p in corpus.pairs if p["pair_id"] in expected]
    for seed, rows in by_seed.items():
        if len(rows) != len(expected) or {r["pair_id"] for r in rows} != expected:
            raise ValueError("Each seed must contain exactly one prediction for every non-development test pair")
        result["seeds"][seed] = {"pooled": evaluate(pairs, rows), "folds": reports[seed]}
        write_jsonl(Path(output).parent / f"pooled_seed_{seed}.jsonl", rows)
    pooled_f1 = {group: {} for group in ("overall", "intra", "cross")}
    for group, metrics in pooled_f1.items():
        for metric in ("direction", "direction_fp_only"):
            scores = [value["pooled"][group][metric]["f1"] for value in result["seeds"].values()]
            valid = [score for score in scores if score is not None]
            metrics[metric] = {"mean": float(np.mean(valid)) if valid else None,
                               "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None}
    result["pooled_f1"] = pooled_f1
    result["pooled_direction_f1_mean"] = pooled_f1["overall"]["direction"]["mean"]
    result["pooled_direction_f1_std"] = pooled_f1["overall"]["direction"]["std"]
    result["test_only"] = any(value["pooled"]["test_only"] for value in result["seeds"].values())
    write_json(output, result)
    return result
