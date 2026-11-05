"""在 verify-local 产物上核验后处理入口，不执行正式研究实验。"""

import argparse
from pathlib import Path

from mode_eci.calibration import calibrate_records, decode_file
from mode_eci.io import read_json, write_json
from mode_eci.pipeline import Experiment
from mode_eci.reporting import compare_files, summarize_files


def load_fixture_experiment(directory):
    root = Path(directory) / "artifacts" / "test_only_bart_slots" / "seed_209" / "fold_0"
    config = read_json(root / "manifest.json")["expert_config"]
    gate_checkpoint = next((root / "gates").glob("G3_*/checkpoint/gate.json"))
    config["gate"] = read_json(gate_checkpoint)["settings"]
    return Experiment(config, "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-dir", default="runs/verification_delivery")
    args = parser.parse_args()
    directory = Path(args.verification_dir).resolve()
    experiment = load_fixture_experiment(directory)
    equal, _ = experiment.predict("test", "equal")
    learned, _ = experiment.predict("test", "learned", "G3")
    for variant in ("no_demo", "shuffle_labels"):
        _, report = experiment.predict("dev", "equal", demo_variant=variant)
        assert report["test_only"] and report["coverage"] == 1
    _, report = experiment.predict("dev", "learned", "G3", top_k=2)
    assert report["coverage"] == 1
    _, report = experiment.predict("dev", "equal", drop="M")
    assert report["coverage"] == 1
    summary = summarize_files(experiment.config, [equal / "predictions.jsonl"], directory / "summary.json")
    comparison = compare_files(experiment.config, equal / "predictions.jsonl", learned / "predictions.jsonl", directory / "bootstrap.json")
    decoded = decode_file(experiment.config, equal / "predictions.jsonl", directory / "paper_decoding", "paper", .5, "S")
    experiment.expert_predictions("inner-0", experiment.test_docs, "test")
    calibration = calibrate_records(experiment.config, experiment.root / "inner-0/predictions/held/records.jsonl",
                                    experiment.root / "inner-0/predictions/test/records.jsonl", directory / "calibration")
    graph_root = directory / "graph/artifacts/test_only_bert_mlm/seed_209/fold_0"
    config = read_json(graph_root / "manifest.json")["expert_config"]
    config["gate"] = read_json(next((graph_root / "gates").glob("G3_*/checkpoint/gate.json")))["settings"]
    graph = Experiment(config, "0")
    _, graph_report = graph.predict("dev", "learned", "G3", demo_variant="shuffle_graph")
    result = {"status": "test_only", "summary": summary["test_only"], "bootstrap": comparison["test_only"],
              "paper_decode": decoded["test_only"], "calibrated_branches": list(calibration["fits"]),
              "shuffle_graph_coverage": graph_report["coverage"], "diagnostics": ["no_demo", "shuffle_labels", "top2", "drop_M"]}
    write_json(directory / "reporting_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
