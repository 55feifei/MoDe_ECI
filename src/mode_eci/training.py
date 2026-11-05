"""小规模与服务器共用的训练循环；按文档隔离数据由调用方负责。"""

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from .experts import collate
from .io import write_json


def seed_everything(seed, cpu_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_name(name):
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device == "cpu":
        raise ValueError("fp16 training requires CUDA")
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict_expert(model, rows, settings):
    model.eval()
    device = next(model.parameters()).device
    output = []
    for start in range(0, len(rows), settings["batch_size"]):
        examples = rows[start:start + settings["batch_size"]]
        batch = move_batch(collate(examples, model.tokenizer), device)
        with autocast_context(str(device), settings["precision"]):
            result = model(batch)
        for index in range(len(examples)):
            detail = {key: value[index].float().cpu().tolist() for key, value in result["detail"].items()}
            output.append({"probabilities": result["probabilities"][index].float().cpu().tolist(),
                           "scores": result["scores"][index].float().cpu().tolist(), "detail": detail})
    return output


def fit(model, rows, tokenizer, settings, seed, directory, loss_fn, validation_fn, save_fn, parameter_groups=None):
    """训练一项任务，按方向 F1、NLL 依次选最佳轮次。

    中断任务从固定种子重跑；完成任务由上层清单恢复。梯度累积按实际样本数
    加权，包括最后一个不足 batch 的累积组，避免它得到额外权重。
    """
    if not rows:
        raise ValueError(f"No training samples: {directory}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, settings["cpu_threads"])
    device = device_name(settings["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters() if parameter_groups is None else parameter_groups,
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    batches_per_epoch = math.ceil(len(rows) / settings["batch_size"])
    updates = math.ceil(batches_per_epoch / settings["accumulation_steps"]) * settings["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * settings["warmup_ratio"]), updates)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["precision"] == "fp16")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    history, best, stale, best_epoch = [], None, 0, 0
    started = time.monotonic()
    for epoch in range(settings["epochs"]):
        model.train()
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        batches = [indices[i:i + settings["batch_size"]] for i in range(0, len(indices), settings["batch_size"])]
        total_loss, optimizer_steps = 0.0, 0
        for start in range(0, len(batches), settings["accumulation_steps"]):
            group = batches[start:start + settings["accumulation_steps"]]
            group_size = sum(len(batch) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for indices_batch in group:
                examples = [rows[i] for i in indices_batch]
                batch = move_batch(collate(examples, tokenizer), device)
                with autocast_context(device, settings["precision"]):
                    loss = loss_fn(model, batch, examples)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in epoch {epoch + 1}")
                scaler.scale(loss * len(examples) / group_size).backward()
                total_loss += float(loss.detach()) * len(examples)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer_steps += 1
        validation = validation_fn(model) if validation_fn is not None else None
        key = (validation["direction_f1"], -validation["nll"]) if validation is not None else (epoch, 0)
        record = {"epoch": epoch + 1, "training_loss": total_loss / len(rows), "validation": validation,
                  "optimizer_steps": optimizer_steps, "last_gradient_norm": float(norm),
                  "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        print(f"{directory.name}: epoch={epoch + 1} loss={record['training_loss']:.5f} validation={validation}", flush=True)
        if best is None or key > best:
            best, stale, best_epoch = key, 0, epoch + 1
            save_fn(model)
        else:
            stale += 1
        write_json(directory / "history.json", history)
        if validation_fn is not None and stale >= settings["patience"]:
            break
    summary = {"status": "complete", "best_epoch": best_epoch, "epochs_run": len(history),
               "training_samples": len(rows), "seconds": time.monotonic() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None}
    write_json(directory / "training.json", summary)
    return summary
