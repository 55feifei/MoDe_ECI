"""单轮预测图扩展：种子模型和 G 专家都按文档交叉拟合。"""

import copy
import random
from collections import deque

import numpy as np
import torch

from .evaluation import validation_metrics
from .experts import Expert
from .io import digest, read_json, read_jsonl, write_json
from .training import fit, predict_expert, seed_everything


class PredictedGraph:
    def __init__(self, corpus, records, settings):
        self.corpus, self.settings = corpus, settings
        self.edges = {}
        for row in records:
            pair = corpus.by_id[row["pair_id"]]
            if "probabilities" in row:
                probabilities = row["probabilities"]
            else:
                probabilities = np.mean([r["probabilities"] for r in row["experts"].values()], axis=0)
            label = int(np.argmax(probabilities))
            if label == 0 or probabilities[label] < settings["threshold"]:
                continue
            events = corpus.documents[pair["doc_id"]]["events"]
            first, second = events[pair["event1"]]["id"], events[pair["event2"]]["id"]
            source, target = (first, second) if label == 1 else (second, first)
            edge = {"source": source, "target": target, "confidence": float(probabilities[label]),
                    "margin": float(probabilities[label] - sorted(probabilities)[-2]), "pair_id": row["pair_id"]}
            self.edges.setdefault(pair["doc_id"], []).append(edge)

    def serialize(self, query, shuffled=False, seed=209):
        # 先删除查询两个方向的边，再搜索邻域，避免通过查询边改变距离。
        targets = {query["e1"]["id"], query["e2"]["id"]}
        edges = [e for e in self.edges.get(query["doc_id"], []) if {e["source"], e["target"]} != targets]
        adjacency = {}
        for edge in edges:
            adjacency.setdefault(edge["source"], set()).add(edge["target"])
            adjacency.setdefault(edge["target"], set()).add(edge["source"])
        distances = {event: 0 for event in targets}
        queue = deque(targets)
        while queue:
            event = queue.popleft()
            if distances[event] >= self.settings["max_hops"]:
                continue
            for neighbor in sorted(adjacency.get(event, [])):
                if neighbor not in distances:
                    distances[neighbor] = distances[event] + 1
                    queue.append(neighbor)
        limit = self.settings["max_hops"]
        local = [e for e in edges if e["source"] in distances and e["target"] in distances
                 and min(distances[e["source"]], distances[e["target"]]) < limit]
        local.sort(key=lambda e: (min(distances[e["source"]], distances[e["target"]]), -e["confidence"], e["source"], e["target"]))
        local = local[:self.settings["max_edges"]]
        if not local:
            return ""
        names = {e["id"]: e["text"] for e in self.corpus.documents[query["doc_id"]]["events"]}
        if shuffled:
            # 对已选边扰乱事件名称，保留边数和置信度用于信息对照。
            values = list(names.values())
            random.Random(int(digest([seed, query["pair_id"]])[:16], 16)).shuffle(values)
            names = dict(zip(names, values))
        return "\n".join(f"event_{e['source']} ({names[e['source']]}) -> event_{e['target']} ({names[e['target']]}), confidence={e['confidence']:.4f}" for e in local)


def seed_experiment(experiment, scope):
    from .pipeline import Experiment
    config = copy.deepcopy(experiment.config)
    config["branches"] = [b for b in config["branches"] if b != "G"]
    config["graph"]["enabled"] = False
    config["output_dir"] = str(experiment.root / "graph_seed")
    config["experiment_id"] = scope
    train, _ = experiment.scope_documents(scope)
    return Experiment(config, experiment.fold, experiment.encoder, training_documents=train)


def graph_texts(experiment, scope, documents, training=False, shuffled=False):
    seed = seed_experiment(experiment, scope)
    if training:
        seed.generate_oof()
        # 复用监督构建时的泄漏检查，但仅向构图器传不带 gold 的原始记录。
        seed.supervised_oof()
        records = list(read_jsonl(seed.root / "oof" / "records.jsonl"))
    else:
        seed.train_experts("final")
        records = seed.expert_predictions("final", documents, "graph_" + digest(sorted(documents))[:12])
    graph = PredictedGraph(experiment.corpus, records, experiment.config["graph"])
    return {pair["pair_id"]: graph.serialize(experiment.corpus.query(pair), shuffled, experiment.config["seed"])
            for pair in experiment.corpus.pairs_for(documents)}


def train_graph_expert(experiment, scope):
    from .pipeline import release
    context = experiment.context(scope)
    train_graphs = graph_texts(experiment, scope, context["train"], training=True)
    validation_graphs = graph_texts(experiment, scope, experiment.dev_docs)
    settings = experiment.config["training"]
    seed_everything(experiment.config["seed"], settings["cpu_threads"])
    model = Expert.create(experiment.config["model"])
    builder = experiment.input_builder(model.tokenizer, model.head)
    rows = experiment.expert_rows(context, context["train"], "G", builder, graphs=train_graphs)
    validation = experiment.expert_rows(context, experiment.dev_docs, "G", builder, graphs=validation_graphs)
    destination = context["directory"] / "experts" / "G"
    if not rows:
        write_json(destination / "training.json", {"status": "complete", "available": False, "training_samples": 0,
                                                   "reason": "No predicted graph evidence in training documents"})
        write_json(destination / "provenance.json", {"train_documents": context["train"], "branch": "G",
                                                    "seed_graph": "document_oof_equal_no_graph", "test_only": experiment.config["test_only"]})
        release(model)
        return

    def loss_fn(current, batch, examples):
        labels = torch.tensor([r["label"] for r in examples], device=batch["input_ids"].device)
        return current(batch, labels)["loss"]

    def validate(current):
        output = predict_expert(current, validation, settings)
        return validation_metrics([r["label"] for r in validation], [r["probabilities"] for r in output])

    fit(model, rows, model.tokenizer, settings, experiment.config["seed"], destination, loss_fn,
        validate if validation else None, lambda current: current.save(destination / "checkpoint"))
    write_json(destination / "provenance.json", {"train_documents": context["train"], "branch": "G",
                                                "seed_graph": "document_oof_equal_no_graph", "test_only": experiment.config["test_only"],
                                                "graph_training_coverage": len(rows) / len(experiment.corpus.pairs_for(context["train"]))})
    release(model)
