"""query-only BERT 门控及 G1/G2/G3 损失。"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .io import checkpoint_fingerprint, read_json, write_json
from .prompts import register_tokens


def masked_weights(logits, mask, temperature=1.0, top_k=None):
    if not torch.all(mask.any(-1)):
        raise ValueError("Each query requires at least one active expert")
    values = (logits / temperature).masked_fill(~mask, -torch.inf)
    weights = values.softmax(-1)
    if top_k is not None:
        # stable=True 在并列时保留固定分支顺序。
        indices = torch.argsort(weights, dim=-1, descending=True, stable=True)[:, :top_k]
        keep = torch.zeros_like(mask).scatter_(1, indices, True)
        weights = weights * keep
        weights = weights / weights.sum(-1, keepdim=True)
    return weights


def teacher_targets(probabilities, labels, mask, temperature):
    """全错样本返回 covered=False；零向量不作为合法教师标签使用。"""
    correct = probabilities.argmax(-1).eq(labels[:, None]) & mask
    covered = correct.any(-1)
    gold = probabilities.gather(2, labels[:, None, None].expand(-1, probabilities.shape[1], 1)).squeeze(-1)
    values = (gold / temperature).masked_fill(~correct, -torch.inf)
    # 全错行先设为有限值，之后清零；避免 softmax(-inf...) 产生 NaN 梯度。
    values = torch.where(covered[:, None], values, torch.zeros_like(values))
    targets = values.softmax(-1) * correct
    return targets, covered


def gate_loss(logits, probabilities, labels, mask, settings):
    probabilities = probabilities.detach()
    weights = masked_weights(logits, mask, settings["temperature"])
    fused = (weights[:, :, None] * probabilities).sum(1)
    mix_loss = -fused.gather(1, labels[:, None]).clamp_min(1e-12).log().mean()
    targets, covered = teacher_targets(probabilities, labels, mask, settings["teacher_temperature"])
    log_weights = (logits / settings["temperature"]).masked_fill(~mask, -torch.inf).log_softmax(-1)
    log_weights = log_weights.masked_fill(~mask, 0)
    teacher_per_row = -(targets * log_weights).sum(-1)
    objective = settings["objective"]
    if objective == "G1":
        return mix_loss
    if objective == "G2":
        if not covered.any():
            raise ValueError("G2 batches must exclude uncovered rows before training")
        return teacher_per_row[covered].mean()
    # G3 的辅助项按全批次平均，等于逐样本 1[C非空] 乘教师损失。
    return mix_loss + settings["teacher_weight"] * teacher_per_row.mean()


class QueryGate(nn.Module):
    def __init__(self, encoder, tokenizer, branches, settings):
        super().__init__()
        self.encoder, self.tokenizer = encoder, tokenizer
        self.branches, self.settings = branches, settings
        self.mlp = nn.Sequential(nn.Linear(encoder.config.hidden_size, settings["hidden_size"]), nn.GELU(),
                                 nn.Dropout(settings["dropout"]), nn.Linear(settings["hidden_size"], len(branches)))

    @classmethod
    def create(cls, branches, settings):
        tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"])
        register_tokens(tokenizer)
        encoder = AutoModel.from_pretrained(settings["name"], revision=settings["revision"])
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        return cls(encoder, tokenizer, branches, settings)

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in self.tokenizer.model_input_names}
        return self.mlp(self.encoder(**inputs).last_hidden_state[:, 0]).float()

    def save(self, directory):
        directory = Path(directory)
        self.encoder.save_pretrained(directory / "encoder")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        torch.save(self.mlp.state_dict(), directory / "mlp.pt")
        write_json(directory / "gate.json", {"branches": self.branches, "settings": self.settings,
                                            "checkpoint_fingerprint": checkpoint_fingerprint(directory)})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = read_json(directory / "gate.json")
        if checkpoint_fingerprint(directory) != meta["checkpoint_fingerprint"]:
            raise ValueError("Gate checkpoint fingerprint differs")
        model = cls(AutoModel.from_pretrained(directory / "encoder"), AutoTokenizer.from_pretrained(directory / "tokenizer"),
                    meta["branches"], meta["settings"])
        model.mlp.load_state_dict(torch.load(directory / "mlp.pt", map_location="cpu", weights_only=True))
        return model
