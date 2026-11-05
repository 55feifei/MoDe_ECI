"""统一的 UTF-8 读写、配置和实验指纹。"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def code_fingerprint():
    return digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


@lru_cache(maxsize=128)
def _asset_digest(path, size, modified_ns):
    return file_digest(path)


def model_identity(settings):
    """远程模型固定 commit；本地模型额外记录实际权重和 tokenizer 内容。"""
    directory = Path(settings["name"])
    if not directory.is_dir():
        if settings["revision"] is None:
            raise ValueError(f"Remote model requires pinned revision: {settings['name']}")
        return {"name": settings["name"], "revision": settings["revision"]}
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".txt", ".model"):
            stat = path.stat()
            files[str(path.relative_to(directory))] = _asset_digest(str(path), stat.st_size, stat.st_mtime_ns)
    return {"path": str(directory.resolve()), "files": files}


def checkpoint_fingerprint(directory):
    """训练权重和 tokenizer 的指纹，不包含存放指纹自身的根元数据。"""
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".pt", ".bin", ".txt", ".model") and path.name not in ("expert.json", "gate.json"):
            stat = path.stat()
            files[str(path.relative_to(directory))] = _asset_digest(str(path), stat.st_size, stat.st_mtime_ns)
    return digest(files)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = next(parent for parent in path.parents if (parent / "pyproject.toml").exists())
    config["data_dir"] = str((root / config["data_dir"]).resolve())
    config["output_dir"] = str((root / config["output_dir"]).resolve())
    for section in ("model", "gate", "retrieval"):
        candidate = root / config[section]["name"]
        if candidate.is_dir():
            config[section]["name"] = str(candidate.resolve())
    for specification in config["input"]["comparison_tokenizers"]:
        candidate = root / specification["name"]
        if candidate.is_dir():
            specification["name"] = str(candidate.resolve())
    if config["retrieval"]["review_file"] is not None:
        config["retrieval"]["review_file"] = str((root / config["retrieval"]["review_file"]).resolve())
    validate_config(config)
    return config


def validate_config(config):
    """仅在配置边界检查会改变实验定义的选项。"""
    choices = {
        "head": (config["model"]["head"], {"bert_mlm", "bert_cls", "bart_relation", "bart_slots"}),
        "rule": (config["rules"]["mode"], {"scope", "lexical"}),
        "retrieval": (config["retrieval"]["strategy"], {"R0", "R1", "R2", "R3"}),
        "pool": (config["retrieval"]["pool"], {"four", "single"}),
        "gate": (config["gate"]["objective"], {"G1", "G2", "G3"}),
        "precision": (config["training"]["precision"], {"fp32", "fp16", "bf16"}),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"Unknown {name}: {value}")
    if config["retrieval"]["demo_count"] not in (2, 3):
        raise ValueError("demo_count must be 2 or 3")
    if config["inner_folds"] < 2 or config["gate"]["teacher_temperature"] <= 0:
        raise ValueError("inner_folds >= 2 and teacher_temperature > 0 required")
    if config["model"]["temperature"] <= 0 or config["gate"]["temperature"] <= 0:
        raise ValueError("Temperatures must be positive")
    branches = config["branches"]
    if not branches or len(branches) != len(set(branches)) or not set(branches) <= {"D1", "D2", "D3", "D4", "C", "S", "M", "G"}:
        raise ValueError("Invalid or duplicate branches")
    if "G" in branches and not config["graph"]["enabled"]:
        raise ValueError("G requires graph.enabled")
    if config["model"]["warmup"] and "S" not in branches:
        raise ValueError("Task warmup requires the S branch")
    if config["training"]["batch_size"] < 1 or config["training"]["accumulation_steps"] < 1:
        raise ValueError("batch_size and accumulation_steps must be positive")
    if config["training"]["epochs"] < 1 or config["gate"]["epochs"] < 1:
        raise ValueError("Training epochs must be positive")
    if not 0 < config["gate"]["validation_fraction"] < 1 or config["gate"]["teacher_weight"] < 0:
        raise ValueError("Gate validation_fraction must be in (0,1) and teacher_weight >= 0")


def claim_run(directory, identity):
    """复用同一实验，拒绝在已有目录中混入另一配置或代码版本。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    record = dict(identity, fingerprint=digest(identity))
    if path.exists() and read_json(path) != record:
        raise ValueError(f"Experiment identity differs; use a new output directory: {directory}")
    write_json(path, record)
    return record
