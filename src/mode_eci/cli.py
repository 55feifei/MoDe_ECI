"""服务器分阶段命令和本机离线验收入口。"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from .data import Corpus, download_esc, prepare_esc
from .io import load_config, read_jsonl, write_json


def parser():
    result = argparse.ArgumentParser(description="MoDE ECI v4 experiments")
    result.add_argument("--config", default="configs/bert_mlm.yaml")
    result.add_argument("--fold", default="0")
    commands = result.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare-data")
    data.add_argument("--raw-dir", default="data/raw")
    data.add_argument("--download", action="store_true")
    data = commands.add_parser("import-data")
    data.add_argument("--source", required=True)
    commands.add_parser("plan")
    audit = commands.add_parser("audit-rules")
    audit.add_argument("--sample", type=int, default=200)
    index = commands.add_parser("build-index")
    index.add_argument("--scope", default="final")
    experts = commands.add_parser("train-experts")
    experts.add_argument("--scope", default="final")
    experts.add_argument("--branches", nargs="+")
    commands.add_parser("oof")
    cache = commands.add_parser("cache-experts")
    cache.add_argument("--scope", default="inner-0")
    cache.add_argument("--split", choices=["held", "dev", "test"], default="held")
    gate = commands.add_parser("train-gate")
    gate.add_argument("--objective", choices=["G1", "G2", "G3"])
    prediction = commands.add_parser("predict")
    prediction.add_argument("--split", choices=["dev", "test"], default="dev")
    prediction.add_argument("--fusion", choices=["equal", "learned", "D", "C", "S", "M"], default="equal")
    prediction.add_argument("--objective", choices=["G1", "G2", "G3"])
    prediction.add_argument("--top-k", type=int, choices=[1, 2])
    prediction.add_argument("--drop", choices=["D1", "D2", "D3", "D4", "C", "S", "M", "G"])
    prediction.add_argument("--demo-variant", choices=["normal", "no_demo", "shuffle_labels", "shuffle_graph"], default="normal")
    run = commands.add_parser("run")
    run.add_argument("--split", choices=["dev", "test"], default="dev")
    run.add_argument("--objectives", nargs="+", choices=["G1", "G2", "G3"], default=["G1", "G2", "G3"])
    verification = commands.add_parser("verify-local")
    verification.add_argument("--output", default="runs/verification")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--first", required=True)
    comparison.add_argument("--second", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--by-topic", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--predictions", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    commands.add_parser("doctor")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--calibration-records", required=True)
    calibration.add_argument("--prediction-records", required=True)
    calibration.add_argument("--output", required=True)
    decode = commands.add_parser("decode")
    decode.add_argument("--predictions", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--mode", choices=["paper", "existence"], required=True)
    decode.add_argument("--threshold", type=float, required=True)
    decode.add_argument("--branch", choices=["S", "C", "M", "D"])
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare-data":
        if args.download:
            download_esc(args.raw_dir)
        output = prepare_esc(args.raw_dir, config["data_dir"], config["seed"])
    elif args.command == "import-data":
        destination = Path(config["data_dir"])
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Data directory must be empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("documents.jsonl", "pairs.jsonl", "splits.json", "statistics.json"):
            shutil.copy2(Path(args.source) / name, destination / name)
        corpus = Corpus(destination)
        output = {"documents": len(corpus.documents), "pairs": len(corpus.pairs), "fingerprint": corpus.fingerprint}
    elif args.command == "verify-local":
        from .verification import verify_local
        output = verify_local(config, args.output)
    elif args.command == "doctor":
        import importlib.metadata
        import torch
        import spacy
        output = {"versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers", "spacy")},
                  "cuda": torch.cuda.is_available(), "spacy_model_installed": spacy.util.is_package(config["rules"]["spacy_model"]),
                  "data_exists": (Path(config["data_dir"]) / "splits.json").exists()}
    elif args.command in ("compare", "summarize"):
        from .reporting import compare_files, summarize_files
        output = compare_files(config, args.first, args.second, args.output, args.by_topic) if args.command == "compare" else summarize_files(config, args.predictions, args.output)
    elif args.command == "calibrate":
        from .calibration import calibrate_records
        output = calibrate_records(config, args.calibration_records, args.prediction_records, args.output, args.fold)
    elif args.command == "decode":
        from .calibration import decode_file
        if args.mode == "paper" and args.branch is None:
            raise ValueError("Paper decoding requires a selected BART slots branch")
        output = decode_file(config, args.predictions, args.output, args.mode, args.threshold, args.branch)
    else:
        from .pipeline import Experiment
        experiment = Experiment(config, args.fold)
        if args.command == "plan":
            output = experiment.plan()
        elif args.command == "build-index":
            context = experiment.context(args.scope)
            output = {"directory": str(context["directory"]), "index": None if context["index"] is None else context["index"].metadata}
        elif args.command == "audit-rules":
            features = experiment.get_features()
            pairs = experiment.corpus.pairs_for(experiment.train_docs)
            random.Random(config["seed"]).shuffle(pairs)
            per_cell = (args.sample + 3) // 4
            counts, selected = {}, []
            for pair in pairs:
                cell = features[pair["pair_id"]]["demo_cell"]
                if counts.get(cell, 0) < per_cell and len(selected) < args.sample:
                    selected.append(pair)
                    counts[cell] = counts.get(cell, 0) + 1
            path = experiment.root / "rule_audit.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["pair_id", "demo_cell", "gold", "lexical_hits", "unresolved", "text", "review_status", "reason"])
                from .features import pair_text
                for pair in selected:
                    f = features[pair["pair_id"]]
                    writer.writerow([pair["pair_id"], f["demo_cell"], pair["label"], ";".join(f["lexical_hits"]), f["unresolved"],
                                     pair_text(experiment.corpus.query(pair)), "", ""])
            output = {"path": str(path), "sampled": len(selected), "counts": counts}
        elif args.command == "train-experts":
            experiment.train_experts(args.scope, args.branches)
            output = {"status": "complete", "scope": args.scope}
        elif args.command == "oof":
            output = {"records": len(experiment.generate_oof())}
        elif args.command == "cache-experts":
            _, held = experiment.scope_documents(args.scope)
            documents = held if args.split == "held" else (experiment.dev_docs if args.split == "dev" else experiment.test_docs)
            output = {"records": len(experiment.expert_predictions(args.scope, documents, args.split))}
        elif args.command == "train-gate":
            experiment.train_gate(args.objective)
            output = {"status": "complete", "objective": args.objective}
        elif args.command == "predict":
            location, report = experiment.predict(args.split, args.fusion, args.objective, args.top_k, args.drop, args.demo_variant)
            output = {"directory": str(location), "overall": report["overall"], "test_only": report["test_only"]}
        elif args.command == "run":
            experiment.run(args.objectives, args.split)
            output = {"status": "complete", "directory": str(experiment.root)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
