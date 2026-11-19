"""独立保留文档上的温度拟合和原论文阈值判决。主实验默认 T=1。"""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Corpus
from .evaluation import evaluate
from .experts import paper_decode
from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .tasks import get_task, records_task


def fit_temperature(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scores / log_temperature.exp(), labels)
        loss.backward()
        return loss

    before = float(F.cross_entropy(scores, labels))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    after = float(F.cross_entropy(scores / temperature, labels))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Non-finite fitted temperature")
    return {"temperature": temperature, "nll_before": before, "nll_after": after, "samples": len(labels)}


def calibrate_records(config, calibration_path, prediction_path, output, fold="0"):
    """单独校准对照：同一冻结专家在独立校准组与评测组上的输出。

    校准组必须属于外层训练池，且未被这些专家训练。输出为重新归一化的
    专家记录；下游门控若使用这些分布，需要在相同校准协议下重新训练。
    """
    corpus = Corpus(config["data_dir"])
    task = get_task(config.get("task", "direction"))
    calibration = list(read_jsonl(calibration_path))
    prediction = list(read_jsonl(prediction_path))
    calibration_meta = read_json(Path(calibration_path).parent / "manifest.json")
    prediction_meta = read_json(Path(prediction_path).parent / "manifest.json")
    for meta in (calibration_meta, prediction_meta):
        task.check(meta)
    records_task(calibration, task)
    records_task(prediction, task)
    if any(calibration_meta[key] != prediction_meta[key] for key in ("train_documents", "scope", "experiment")):
        raise ValueError("Calibration and evaluation require the same frozen expert source")
    calibration_docs = {r["doc_id"] for r in calibration}
    evaluation_docs = {r["doc_id"] for r in prediction}
    if calibration_docs & (set(calibration_meta["train_documents"]) | evaluation_docs):
        raise ValueError("Calibration documents overlap training or evaluation")
    if evaluation_docs & set(prediction_meta["train_documents"]):
        raise ValueError("Evaluation documents overlap expert training")
    outer_pool = set(corpus.documents_for(fold, "train"))
    if not calibration_docs <= outer_pool:
        raise ValueError("Temperature fitting requires training-side held-out documents")
    fitted = {}
    for branch in calibration_meta["branches"]:
        rows = [r for r in calibration if branch in r["experts"]]
        if not rows:
            raise ValueError(f"No calibration coverage for {branch}")
        scores = [r["experts"][branch]["scores"] for r in rows]
        for values in scores:
            task.check_probabilities(values)
        fitted[branch] = fit_temperature(scores, [task.label(corpus.by_id[r["pair_id"]]["label"]) for r in rows])
    result = copy.deepcopy(prediction)
    for row in result:
        for branch, expert in row["experts"].items():
            expert["probabilities"] = (torch.tensor(expert["scores"]) / fitted[branch]["temperature"]).softmax(-1).tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "records.jsonl", result)
    manifest = {**task.metadata(), "fits": fitted, "calibration_documents": sorted(calibration_docs), "evaluation_documents": sorted(evaluation_docs),
                "source": prediction_meta, "calibration_fingerprint": digest(calibration)}
    write_json(output / "calibration.json", manifest)
    return manifest


def decode_file(config, path, output, mode, threshold, branch=None):
    """阈值应在开发阶段固定；本命令不在测试集搜索阈值。"""
    corpus = Corpus(config["data_dir"])
    predictions = copy.deepcopy(list(read_jsonl(path)))
    task = records_task(predictions, config.get("task", "direction"))
    if mode == "paper" and task.name == "existence":
        raise ValueError("Paper direction decoding is unavailable for existence models")
    if mode not in ("paper", "existence"):
        raise ValueError(f"Unknown decoding mode: {mode}")
    if mode == "existence" and not 0 <= threshold <= 1:
        raise ValueError("Existence threshold must be in [0, 1]")
    for row in predictions:
        if mode == "paper":
            selected = row["features"]["demo_cell"] if branch == "D" else branch
            expert = row["experts"][selected]
            slots = torch.tensor([expert["detail"]["slot_token_probabilities"]])
            row["label"] = int(paper_decode(slots, threshold)[0])
            row["probabilities"] = expert["probabilities"]
            row["weights"] = [float(b == selected) for b in row["branches"]]
        else:
            q = row["probabilities"]
            if task.name == "existence":
                row["label"] = int(q[1] >= threshold)
            else:
                row["label"] = (1 if q[1] >= q[2] else 2) if q[1] + q[2] >= threshold else 0
        row["decision"] = {"mode": mode, "threshold": threshold, "branch": branch}
    ids = {r["pair_id"] for r in predictions}
    pairs = [p for p in corpus.pairs if p["pair_id"] in ids]
    report = evaluate(pairs, predictions, task)
    directory = Path(output)
    write_jsonl(directory / "predictions.jsonl", predictions)
    write_json(directory / "metrics.json", report)
    write_json(directory / "decision.json", {**task.metadata(), "mode": mode, "threshold": threshold, "branch": branch, "source": str(path)})
    return report
