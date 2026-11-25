"""离线复用权重，完成一个外层折的 D_all、多专家、G3 和 dev/test 验收。"""

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# 解释器可来自相邻项目，但执行的必须是本仓库源码。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from mode_eci.io import load_config, read_json, write_json
from mode_eci.pipeline import Experiment
from evaluate_demo_specialists import evaluate_specialists


def audit_training(experiment):
    stages = []
    for scope in [f"inner-{i}" for i in range(len(experiment.groups))] + ["final"]:
        train, held = experiment.scope_documents(scope)
        experts = experiment.root / scope / "experts"
        expected = {p["pair_id"] for p in experiment.corpus.pairs_for(train)}
        parent = experts / "D_all"
        provenance = read_json(parent / "provenance.json")
        if set(provenance["training_pair_ids"]) != expected:
            raise ValueError(f"D_all training coverage differs: {scope}")
        if set(provenance["train_documents"]) & (set(held) | set(experiment.dev_docs) | set(experiment.test_docs)):
            raise ValueError(f"D_all document leakage: {scope}")
        parent_fingerprint = read_json(parent / "checkpoint" / "expert.json")["checkpoint_fingerprint"]
        for branch in ["D_all"] + experiment.config["branches"]:
            directory = experts / branch
            info = read_json(directory / "provenance.json")
            summary = read_json(directory / "training.json")
            checkpoint = directory / "checkpoint" / "expert.json"
            if summary["status"] != "complete" or (summary.get("available", True) and not checkpoint.exists()):
                raise ValueError(f"Incomplete expert: {directory}")
            if branch in ("D1", "D2", "D3", "D4"):
                if info["initialization_fingerprint"] != parent_fingerprint:
                    raise ValueError(f"Specialist was not initialized from D_all: {directory}")
                if info["training_cells"][branch] != summary["training_samples"]:
                    raise ValueError(f"Specialist contains out-of-cell queries: {directory}")
            stages.append(dict(scope=scope, branch=branch, **summary))
    experiment.supervised_oof()  # 同时校验 OOF 完整覆盖和文档隔离。
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/roberta_mlm_dall_seed3407_gpu.yaml"))
    parser.add_argument("--fold", default="0")
    parser.add_argument("--output", default=str(ROOT / "runs/roberta_dall_seed3407_gpu_e15_p5_focal2"))
    parser.add_argument("--isolate-stages", action="store_true",
                        help="Run inner-fold training, held-out inference, OOF and gate training in separate processes.")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"status": "running", "python": sys.executable, "config": str(Path(args.config).resolve()),
             "fold": args.fold, "offline": True, "isolated_stages": args.isolate_stages,
             "versions": {name: importlib.metadata.version(name) for name in
                          ("torch", "transformers", "sentence-transformers", "spacy")}}

    def stage(name):
        state.update(stage=name, elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", state)
        print(f"STAGE {name}", flush=True)

    def run_stage(*command):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, "-u", "-m", "mode_eci", "--config", state["config"],
                        "--fold", args.fold, *command], cwd=ROOT, env=env, check=True)

    try:
        config = load_config(args.config)
        if not config["model"].get("demo_pretraining", False):
            raise ValueError("This runner requires model.demo_pretraining=true")
        if config["training"]["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA training is unavailable")
        state["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(output / "effective_config.json", config)
        stage("plan")
        experiment = Experiment(config, args.fold)
        state["root"] = str(experiment.root)
        write_json(output / "plan.json", experiment.plan())
        stage("train_final_demo_experts")
        experiment.train_experts("final", ["D1", "D2", "D3", "D4"])
        early_reports = {}
        for split in ("dev", "test"):
            for shared in (True, False):
                mode = "D_all" if shared else "D_specialists"
                stage(f"early_evaluate_{split}_{mode}")
                directory, report = evaluate_specialists(experiment, split, shared)
                early_reports[f"{split}_{mode}"] = {"directory": str(directory), "metrics": report}
                write_json(output / "demo_results.json", early_reports)
        stage("train_remaining_final_experts")
        experiment.train_experts("final")
        if args.isolate_stages:
            experiment.contexts.clear()
            experiment.encoder = None
            for number in range(len(experiment.groups)):
                scope = f"inner-{number}"
                if (experiment.root / scope / "predictions" / "held" / "complete.json").is_file():
                    continue
                stage(f"train_experts_{scope}")
                run_stage("train-experts", "--scope", scope)
                stage(f"cache_held_{scope}")
                run_stage("cache-experts", "--scope", scope, "--split", "held")
        stage("generate_oof")
        if args.isolate_stages:
            run_stage("oof")
        else:
            experiment.generate_oof()
        objective = config["gate"]["objective"]
        stage(f"train_gate_{objective}")
        if args.isolate_stages:
            run_stage("train-gate", "--objective", objective)
        else:
            experiment.train_gate(objective)
        reports = {}
        for split in ("dev", "test"):
            for fusion in ("D", "equal", "learned"):
                stage(f"evaluate_{split}_{fusion}")
                directory, report = experiment.predict(split, fusion, objective if fusion == "learned" else None)
                reports[f"{split}_{fusion}"] = {"directory": str(directory), "metrics": report}
        stage("audit_training")
        stages = audit_training(experiment)
        write_json(output / "training_audit.json", stages)
        write_json(output / "results.json", reports)
        state.update(status="complete", stage="complete", elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", state)
        print(f"COMPLETE {output}", flush=True)
    except Exception:
        state.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
