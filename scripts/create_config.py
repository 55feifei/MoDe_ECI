"""生成独立消融配置，不修改原配置，也不覆盖已有文件。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/bert_mlm.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strategy", choices=["R0", "R1", "R2", "R3"])
    parser.add_argument("--pool", choices=["four", "single"])
    parser.add_argument("--demo-count", type=int, choices=[2, 3])
    parser.add_argument("--rule", choices=["scope", "lexical"])
    parser.add_argument("--branches", nargs="+")
    parser.add_argument("--review-file")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--teacher-temperature", type=float)
    parser.add_argument("--teacher-weight", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    for flag, section, name in [("seed", None, "seed"), ("strategy", "retrieval", "strategy"), ("pool", "retrieval", "pool"),
                                ("demo_count", "retrieval", "demo_count"), ("rule", "rules", "mode"),
                                ("branches", None, "branches"), ("review_file", "retrieval", "review_file"),
                                ("learning_rate", "training", "learning_rate"), ("teacher_temperature", "gate", "teacher_temperature"),
                                ("teacher_weight", "gate", "teacher_weight")]:
        value = getattr(args, flag)
        if value is not None:
            target = config if section is None else config[section]
            target[name] = value
    if args.graph:
        config["graph"]["enabled"] = True
        if "G" not in config["branches"]:
            config["branches"].append("G")
    if args.warmup:
        config["model"]["warmup"] = True
    definition = copy.deepcopy(config)
    # 同一方法的不同种子仍用同一 experiment_id，输出由 seed 子目录隔离。
    definition.pop("seed")
    suffix = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:8]
    config["experiment_id"] += "_" + suffix
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(output)


if __name__ == "__main__":
    main()
