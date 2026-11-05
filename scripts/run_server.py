"""冻结配置后顺序运行五折多种子；完成阶段自动复用。"""

import argparse
import copy
import os

from mode_eci.io import load_config
from mode_eci.pipeline import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bert_mlm.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[209, 42, 3407])
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--stage", choices=["development", "test"], default="development")
    args = parser.parse_args()
    # CUDA GEMM 的确定性工作区在首次 CUDA 操作前配置。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base = load_config(args.config)
    for seed in args.seeds:
        for fold in args.folds:
            config = copy.deepcopy(base)
            config["seed"] = seed
            experiment = Experiment(config, fold)
            if args.stage == "development":
                experiment.run(split="dev")
            else:
                # 测试阶段消费已完成的训练产物，不根据测试成绩选择配置。
                experiment.predict("test", "equal")
                for objective in ("G1", "G2", "G3"):
                    experiment.predict("test", "learned", objective)


if __name__ == "__main__":
    main()
