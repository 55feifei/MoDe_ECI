"""统一三类接口的 BERT MLM、事件 MLP、BART 关系和角色槽位专家。"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .io import checkpoint_fingerprint, read_json, write_json
from .prompts import RELATION_TOKENS, register_tokens


def collate(rows, tokenizer):
    """动态 padding；事件 mask 的零填充与 attention mask 保持一致。"""
    features = [{k: v for k, v in row["encoded"].items() if k in tokenizer.model_input_names} for row in rows]
    batch = dict(tokenizer.pad(features, padding=True, return_tensors="pt"))
    length = batch["input_ids"].shape[1]
    for name in ("event1_mask", "event2_mask"):
        batch[name] = torch.tensor([row["encoded"][name] + [0] * (length - len(row["encoded"][name])) for row in rows], dtype=torch.float32)
    if "mask_position" in rows[0]["encoded"]:
        batch["mask_position"] = torch.tensor([row["encoded"]["mask_position"] for row in rows])
    return batch


class Expert(nn.Module):
    def __init__(self, backbone, tokenizer, head, temperature=1.0, slot_relation_weight=0.0):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head
        self.temperature = temperature
        self.slot_relation_weight = slot_relation_weight
        self.relation_ids = tokenizer.convert_tokens_to_ids(RELATION_TOKENS)
        if head == "bert_cls":
            hidden = backbone.config.hidden_size
            self.classifier = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 3))
        if head.startswith("bart"):
            template = "In this context, <SLOT> causes <SLOT>." if head == "bart_slots" else "Relation: <SLOT>."
            target_template = tokenizer.encode(template, add_special_tokens=True)
            self.slot_positions = [i for i, token in enumerate(target_template) if token == tokenizer.convert_tokens_to_ids("<SLOT>")]
            # 手工右移的是全样本相同的占位模板。gold 仅用于 loss，不进入 decoder。
            decoder_ids = [backbone.config.decoder_start_token_id] + target_template[:-1]
            self.register_buffer("fixed_decoder_ids", torch.tensor(decoder_ids), persistent=False)
            self.role_ids = tokenizer.convert_tokens_to_ids(["<None>", "<E1>", "<E2>"])

    @classmethod
    def create(cls, settings):
        tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"])
        register_tokens(tokenizer)
        head = settings["head"]
        loader = AutoModelForMaskedLM if head == "bert_mlm" else (AutoModel if head == "bert_cls" else AutoModelForSeq2SeqLM)
        backbone = loader.from_pretrained(settings["name"], revision=settings["revision"])
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        if settings["gradient_checkpointing"]:
            backbone.gradient_checkpointing_enable()
            backbone.config.use_cache = False
        return cls(backbone, tokenizer, head, settings["temperature"], settings["slot_relation_weight"])

    def forward(self, batch, labels=None):
        inputs = {k: v for k, v in batch.items() if k in self.tokenizer.model_input_names}
        if self.head == "bert_mlm":
            logits = self.backbone(**inputs).logits
            rows = torch.arange(logits.shape[0], device=logits.device)
            scores = logits[rows, batch["mask_position"]][:, self.relation_ids].float()
            detail = {"relation_logits": scores}
            loss = None if labels is None else F.cross_entropy(scores, labels)
        elif self.head == "bert_cls":
            hidden = self.backbone(**inputs).last_hidden_state
            means = []
            for name in ("event1_mask", "event2_mask"):
                weights = batch[name].unsqueeze(-1)
                means.append((hidden * weights).sum(1) / weights.sum(1).clamp_min(1))
            scores = self.classifier(torch.cat(means, dim=-1)).float()
            detail = {"relation_logits": scores}
            loss = None if labels is None else F.cross_entropy(scores, labels)
        else:
            size = inputs["input_ids"].shape[0]
            decoder = self.fixed_decoder_ids.unsqueeze(0).expand(size, -1)
            logits = self.backbone(**inputs, decoder_input_ids=decoder, use_cache=False).logits[:, self.slot_positions].float()
            if self.head == "bart_relation":
                scores = logits[:, 0, self.relation_ids]
                detail = {"relation_logits": scores}
                loss = None if labels is None else F.cross_entropy(scores, labels)
            else:
                log_probs = logits.log_softmax(-1)
                pairs = torch.tensor([[self.role_ids[0], self.role_ids[0]], [self.role_ids[1], self.role_ids[2]],
                                      [self.role_ids[2], self.role_ids[1]]], device=logits.device)
                scores = log_probs[:, 0, pairs[:, 0]] + log_probs[:, 1, pairs[:, 1]]
                detail = {"slot_token_probabilities": log_probs[:, :, self.role_ids].exp(), "combination_scores": scores}
                loss = None
                if labels is not None:
                    targets = pairs[labels]
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                    loss = loss + self.slot_relation_weight * F.cross_entropy(scores, labels)
        return {"probabilities": (scores / self.temperature).softmax(-1), "scores": scores, "loss": loss, "detail": detail}

    def save(self, directory):
        directory = Path(directory)
        self.backbone.save_pretrained(directory / "backbone")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        if self.head == "bert_cls":
            torch.save(self.classifier.state_dict(), directory / "classifier.pt")
        write_json(directory / "expert.json", {"head": self.head, "temperature": self.temperature,
                                              "slot_relation_weight": self.slot_relation_weight,
                                              "relation_ids": self.relation_ids, "checkpoint_fingerprint": checkpoint_fingerprint(directory)})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = read_json(directory / "expert.json")
        if checkpoint_fingerprint(directory) != meta["checkpoint_fingerprint"]:
            raise ValueError("Expert checkpoint fingerprint differs")
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        loader = AutoModelForMaskedLM if meta["head"] == "bert_mlm" else (AutoModel if meta["head"] == "bert_cls" else AutoModelForSeq2SeqLM)
        model = cls(loader.from_pretrained(directory / "backbone"), tokenizer, meta["head"],
                    meta["temperature"], meta["slot_relation_weight"])
        if model.relation_ids != meta["relation_ids"]:
            raise ValueError("Relation token mapping changed")
        if meta["head"] == "bert_cls":
            model.classifier.load_state_dict(torch.load(directory / "classifier.pt", map_location="cpu", weights_only=True))
        return model


def paper_decode(slot_probabilities, threshold):
    """原论文独立判决，槽位 token 顺序为 None、E1、E2。"""
    none = slot_probabilities[:, 0, 0] + slot_probabilities[:, 1, 0]
    forward = slot_probabilities[:, 0, 1] + slot_probabilities[:, 1, 2]
    backward = slot_probabilities[:, 0, 2] + slot_probabilities[:, 1, 1]
    direction = torch.where(forward >= backward, 1, 2)
    return torch.where(none > threshold, 0, direction)
