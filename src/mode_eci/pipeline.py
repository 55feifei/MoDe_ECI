"""按文档交叉拟合、专家训练、折外门控与最终预测的实验编排。"""

import copy
import gc
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import Corpus, LABELS
from .evaluation import evaluate, validation_metrics
from .experts import Expert, collate
from .features import RuleRouter, active_mask
from .gate import QueryGate, gate_loss, masked_weights, teacher_targets
from .io import claim_run, code_fingerprint, digest, model_identity, read_json, read_jsonl, write_json, write_jsonl
from .prompts import InputBuilder, register_tokens
from .retrieval import DemoIndex, HashEncoder, MPNetEncoder
from .training import device_name, fit, move_batch, predict_expert, seed_everything


def document_groups(corpus, documents, count, seed):
    """训练侧按文档分组，贪心平衡正例数、文档数及主题覆盖。"""
    if len(documents) < count:
        raise ValueError("Not enough documents for cross-fitting")
    randomizer = random.Random(seed)
    order = list(documents)
    randomizer.shuffle(order)
    positives = {doc: 0 for doc in documents}
    sizes = {doc: 0 for doc in documents}
    for pair in corpus.pairs_for(documents):
        sizes[pair["doc_id"]] += 1
        positives[pair["doc_id"]] += pair["label"] != 0
    order.sort(key=lambda doc: -positives[doc])
    groups, totals, lengths = [[] for _ in range(count)], [0] * count, [0] * count
    topics = [set() for _ in range(count)]
    for doc in order:
        topic = corpus.documents[doc]["topic_id"]
        target = min(range(count), key=lambda i: (totals[i], lengths[i], topic in topics[i], len(groups[i]), i))
        groups[target].append(doc)
        totals[target] += positives[doc]
        lengths[target] += sizes[doc]
        topics[target].add(topic)
    if any(not group for group in groups):
        raise ValueError("Empty document group")
    return [sorted(group) for group in groups]


def release(model):
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Experiment:
    def __init__(self, config, fold="0", encoder=None, training_documents=None):
        self.config = config
        self.fold = str(fold)
        self.corpus = Corpus(config["data_dir"])
        if self.corpus.test_only and not config["test_only"]:
            raise ValueError("Synthetic corpus requires test_only=true")
        for section in ("model", "gate", "retrieval"):
            source = Path(config[section]["name"]) / "source.json"
            if source.exists() and read_json(source).get("test_only", False) and not config["test_only"]:
                raise ValueError("Tiny models are test_only")
        self.train_docs = self.corpus.documents_for(fold, "train")
        if training_documents is not None:
            if not set(training_documents) <= set(self.train_docs):
                raise ValueError("Nested training documents must be within outer training set")
            self.train_docs = sorted(training_documents)
        self.dev_docs = self.corpus.documents_for(fold, "dev")
        self.test_docs = self.corpus.documents_for(fold, "test")
        self.groups = document_groups(self.corpus, self.train_docs, config["inner_folds"], config["seed"])
        self.root = Path(config["output_dir"]) / config["experiment_id"] / f"seed_{config['seed']}" / f"fold_{fold}"
        # 门控目标可在同一组冻结专家上比较。各门控另有完整配置清单。
        expert_config = {key: value for key, value in config.items() if key != "gate"}
        claim_run(self.root, {"expert_config": expert_config, "code": code_fingerprint(), "corpus": self.corpus.fingerprint,
                              "train_documents": self.train_docs, "dev_documents": self.dev_docs,
                              "test_documents": self.test_docs, "inner_groups": self.groups, "test_only": config["test_only"],
                              "assets": {section: model_identity(config[section]) for section in ("model", "gate", "retrieval")
                                         if section != "retrieval" or not config["test_only"]}})
        self.router = RuleRouter(config["rules"]["mode"], config["rules"]["spacy_model"])
        self.encoder = encoder
        self.features = None
        self.contexts = {}

    def plan(self):
        return {"root": str(self.root), "outer_fold": self.fold, "inner_groups": self.groups,
                "expert_training_jobs": (len(self.groups) + 1) * len(self.config["branches"]),
                "train_pairs": len(self.corpus.pairs_for(self.train_docs)), "dev_pairs": len(self.corpus.pairs_for(self.dev_docs)),
                "test_pairs": len(self.corpus.pairs_for(self.test_docs)), "branches": self.config["branches"],
                "test_only": self.config["test_only"]}

    def get_features(self):
        if self.features is None:
            path = self.root / "features.jsonl"
            if path.exists():
                records = list(read_jsonl(path))
            else:
                records = [{"pair_id": p["pair_id"], "features": self.router(self.corpus.query(p))} for p in self.corpus.pairs]
                write_jsonl(path, records)
                write_json(self.root / "rule_manifest.json", {"settings": self.config["rules"], "parser": self.router.parser_version})
            self.features = {r["pair_id"]: r["features"] for r in records}
        return self.features

    def get_encoder(self):
        if self.encoder is None:
            self.encoder = HashEncoder() if self.config["test_only"] else MPNetEncoder(self.config["retrieval"])
        return self.encoder

    def scope_documents(self, scope):
        if scope == "final":
            return self.train_docs, []
        number = int(scope.removeprefix("inner-"))
        held = self.groups[number]
        train = sorted(set(self.train_docs) - set(held))
        return train, held

    def context(self, scope):
        if scope in self.contexts:
            return self.contexts[scope]
        train, held = self.scope_documents(scope)
        directory = self.root / scope
        claim_run(directory, {"train_documents": train, "held_documents": held,
                              "experiment": read_json(self.root / "manifest.json")["fingerprint"]})
        index = None
        if any(b.startswith("D") for b in self.config["branches"]):
            location = directory / "index"
            if (location / "metadata.json").exists():
                index = DemoIndex.load(location)
                if index.metadata["train_documents"] != sorted(train):
                    raise ValueError("Index training documents differ")
            else:
                index = DemoIndex.build(self.corpus, train, self.router, self.get_encoder(), self.config["retrieval"], self.config["test_only"])
                index.save(location)
        context = {"directory": directory, "train": train, "held": held, "index": index}
        self.contexts[scope] = context
        return context

    def retrieval_plans(self, context, documents):
        if context["index"] is None:
            return {}
        key = digest(sorted(documents))[:16]
        path = context["directory"] / "retrieval" / f"{key}.jsonl"
        if path.exists():
            rows = list(read_jsonl(path))
        else:
            pairs = self.corpus.pairs_for(documents)
            queries = [self.corpus.query(p) for p in pairs]
            features = self.get_features()
            rows = context["index"].retrieve_many(queries, [features[p["pair_id"]] for p in pairs], self.get_encoder(),
                                                  self.config["retrieval"], self.config["seed"])
            write_jsonl(path, rows)
        return {r["query_id"]: r for r in rows}

    def input_builder(self, tokenizer, head):
        tokenizers = [tokenizer]
        for specification in self.config["input"]["comparison_tokenizers"]:
            other = AutoTokenizer.from_pretrained(specification["name"], revision=specification["revision"])
            register_tokens(other)
            tokenizers.append(other)
        return InputBuilder(tokenizers, head, self.config["input"])

    def expert_rows(self, context, documents, branch, builder, demo_variant="normal", graphs=None):
        features = self.get_features()
        pairs = self.corpus.pairs_for(documents)
        if branch.startswith("D"):
            pairs = [p for p in pairs if features[p["pair_id"]]["demo_cell"] == branch]
        plans = self.retrieval_plans(context, documents) if branch.startswith("D") else {}
        rows = []
        for pair in pairs:
            query = self.corpus.query(pair)
            feature = features[pair["pair_id"]]
            demos, plan = [], None
            if branch.startswith("D"):
                plan = plans[pair["pair_id"]]
                demos = [context["index"].by_id[d["pair_id"]] for d in plan["demos"]]
                if demo_variant == "no_demo":
                    demos = []
            graph_text = "" if graphs is None else graphs[pair["pair_id"]]
            if branch == "G" and not graph_text:
                continue
            labels = None
            if demo_variant == "shuffle_labels":
                labels = [(d["label"] + 1) % 3 for d in demos]
            built = builder.build(query, branch, feature, demos, graph_text, labels)
            if branch == "G" and built["coverage"]["graph_edges_retained"] == 0:
                continue
            rows.append(dict(built, pair_id=pair["pair_id"], label=pair["label"], retrieval=plan))
        return rows

    def train_experts(self, scope="final", branches=None):
        context = self.context(scope)
        branches = self.config["branches"] if branches is None else branches
        for branch in branches:
            if branch not in self.config["branches"]:
                raise ValueError(f"Branch not configured: {branch}")
            destination = context["directory"] / "experts" / branch
            if (destination / "training.json").exists() and read_json(destination / "training.json")["status"] == "complete":
                continue
            if branch == "G":
                self.train_graph_expert(scope)
                continue
            warmup = context["directory"] / "warmup" / "checkpoint"
            if self.config["model"]["warmup"] and branch != "S":
                self.train_warmup(scope)
            seed_everything(self.config["seed"], self.config["training"]["cpu_threads"])
            if self.config["model"]["warmup"] and branch != "S":
                model = Expert.load(warmup)
            else:
                model = Expert.create(self.config["model"])
            builder = self.input_builder(model.tokenizer, model.head)
            rows = self.expert_rows(context, context["train"], branch, builder)
            validation = self.expert_rows(context, self.dev_docs, branch, builder)

            def loss_fn(current, batch, examples):
                labels = torch.tensor([r["label"] for r in examples], device=batch["input_ids"].device)
                return current(batch, labels)["loss"]

            def validate(current):
                predictions = predict_expert(current, validation, self.config["training"])
                return validation_metrics([r["label"] for r in validation], [r["probabilities"] for r in predictions])

            fit(model, rows, model.tokenizer, self.config["training"], self.config["seed"], destination, loss_fn,
                validate if validation else None, lambda current: current.save(destination / "checkpoint"))
            write_json(destination / "provenance.json", {"train_documents": context["train"], "branch": branch,
                                                        "test_only": self.config["test_only"]})
            release(model)
            del model

    def train_warmup(self, scope):
        """共同任务预热仅使用当前 scope 的训练文档。"""
        context = self.context(scope)
        destination = context["directory"] / "warmup"
        if (destination / "training.json").exists():
            return
        # S 就是相同任务的无 demo 预热，不引入另一份数据或额外标签。
        self.train_experts(scope, ["S"])
        import shutil
        shutil.copytree(context["directory"] / "experts" / "S" / "checkpoint", destination / "checkpoint", dirs_exist_ok=True)
        write_json(destination / "training.json", read_json(context["directory"] / "experts" / "S" / "training.json"))

    def expert_predictions(self, scope, documents, name, demo_variant="normal", graphs=None):
        context = self.context(scope)
        directory = context["directory"] / "predictions" / name
        identity = {"train_documents": context["train"], "prediction_documents": sorted(documents), "demo_variant": demo_variant,
                    "scope": scope, "branches": self.config["branches"], "test_only": self.config["test_only"],
                    "experiment": read_json(self.root / "manifest.json")["fingerprint"],
                    "index": None if context["index"] is None else context["index"].metadata["fingerprint"]}
        claim_run(directory, identity)
        path = directory / "records.jsonl"
        if (directory / "complete.json").exists():
            records = list(read_jsonl(path))
            if digest(records) != read_json(directory / "complete.json")["records_sha256"]:
                raise ValueError("Expert prediction fingerprint differs")
            return records
        if "G" in self.config["branches"] and graphs is None:
            from .graph import graph_texts
            graphs = graph_texts(self, scope, documents, shuffled=demo_variant == "shuffle_graph")
        combined = {p["pair_id"]: {"pair_id": p["pair_id"], "doc_id": p["doc_id"], "features": self.get_features()[p["pair_id"]],
                                    "experts": {}, "source_scope": scope, "test_only": self.config["test_only"]}
                    for p in self.corpus.pairs_for(documents)}
        for branch in self.config["branches"]:
            destination = context["directory"] / "experts" / branch
            training_record = read_json(destination / "training.json")
            if training_record["status"] != "complete":
                raise ValueError(f"Incomplete expert: {destination}")
            if branch == "G" and training_record.get("available") is False:
                continue
            model = Expert.load(destination / "checkpoint").to(device_name(self.config["training"]["device"]))
            builder = self.input_builder(model.tokenizer, model.head)
            rows = self.expert_rows(context, documents, branch, builder, demo_variant, graphs)
            predictions = predict_expert(model, rows, self.config["training"])
            for row, prediction in zip(rows, predictions):
                combined[row["pair_id"]]["experts"][branch] = dict(prediction, text=row["text"], coverage=row["coverage"], retrieval=row["retrieval"])
            release(model)
            del model
        records = list(combined.values())
        write_jsonl(path, records)
        write_json(directory / "complete.json", {"count": len(records), "records_sha256": digest(records)})
        return records

    def generate_oof(self):
        records = []
        provenance = []
        for number, held in enumerate(self.groups):
            scope = f"inner-{number}"
            self.train_experts(scope)
            generated = self.expert_predictions(scope, held, "held")
            train, _ = self.scope_documents(scope)
            if set(train) & set(held):
                raise ValueError("OOF expert training leakage")
            provenance.append({"scope": scope, "train_documents": train, "held_documents": held})
            records.extend(generated)
        expected = {p["pair_id"] for p in self.corpus.pairs_for(self.train_docs)}
        if len(records) != len(expected) or {r["pair_id"] for r in records} != expected:
            raise ValueError("OOF predictions must cover every training pair exactly once")
        # 此文件没有 gold；只有门控监督构建步骤才读取 corpus 的标签。
        write_jsonl(self.root / "oof" / "records.jsonl", records)
        write_json(self.root / "oof" / "provenance.json", {"sources": provenance, "count": len(records),
                                                           "records_fingerprint": digest(records)})
        return records

    def supervised_oof(self):
        records = list(read_jsonl(self.root / "oof" / "records.jsonl"))
        provenance = read_json(self.root / "oof" / "provenance.json")
        if digest(records) != provenance["records_fingerprint"]:
            raise ValueError("OOF file fingerprint differs")
        sources = {r["scope"]: r for r in provenance["sources"]}
        expected = {p["pair_id"] for p in self.corpus.pairs_for(self.train_docs)}
        if len(records) != len(expected) or {r["pair_id"] for r in records} != expected:
            raise ValueError("OOF coverage differs from outer training set")
        for row in records:
            source = sources[row["source_scope"]]
            if row["doc_id"] in source["train_documents"] or row["doc_id"] not in source["held_documents"]:
                raise ValueError("OOF document leakage")
            if not set(source["train_documents"]) <= set(self.train_docs):
                raise ValueError("OOF source outside outer training set")
        return [dict(r, label=self.corpus.by_id[r["pair_id"]]["label"]) for r in records]

    def gate_rows(self, records, builder):
        rows = []
        branches = self.config["branches"]
        for record in records:
            query = self.corpus.query(self.corpus.by_id[record["pair_id"]])
            built = builder.build(query, "gate", record["features"])
            probabilities = [record["experts"][b]["probabilities"] if b in record["experts"] else [0.0] * 3 for b in branches]
            mask = [b in record["experts"] for b in branches]
            expected = active_mask(record["features"]["demo_cell"], branches, "G" in record["experts"])
            if mask != expected or not any(mask):
                raise ValueError(f"Missing active expert: {record['pair_id']}")
            rows.append(dict(record, **built, expert_probabilities=probabilities, active_mask=mask))
        return rows

    def gate_settings(self, objective=None):
        settings = copy.deepcopy(self.config["gate"])
        if objective is not None:
            settings["objective"] = objective
        return settings

    def gate_directory(self, objective=None):
        settings = self.gate_settings(objective)
        return self.root / "gates" / f"{settings['objective']}_{digest(settings)[:12]}"

    def train_gate(self, objective=None):
        settings = self.gate_settings(objective)
        directory = self.gate_directory(objective)
        claim_run(directory, {"settings": settings, "oof": read_json(self.root / "oof" / "provenance.json"),
                              "experiment": read_json(self.root / "manifest.json")["fingerprint"]})
        if (directory / "refit" / "training.json").exists():
            return
        records = self.supervised_oof()
        seed_everything(self.config["seed"], self.config["training"]["cpu_threads"])
        model = QueryGate.create(self.config["branches"], settings)
        builder = InputBuilder([model.tokenizer], "bert_cls", self.config["input"])
        rows = self.gate_rows(records, builder)
        groups = document_groups(self.corpus, self.train_docs, max(2, min(len(self.train_docs), round(1 / settings["validation_fraction"]))), self.config["seed"] + 1)
        validation_docs = set(groups[0])
        train = [r for r in rows if r["doc_id"] not in validation_docs]
        validation = [r for r in rows if r["doc_id"] in validation_docs]
        training_settings = dict(self.config["training"], learning_rate=settings["learning_rate"], epochs=settings["epochs"], patience=settings["patience"])
        targets, covered = teacher_targets(torch.tensor([r["expert_probabilities"] for r in rows]),
                                            torch.tensor([r["label"] for r in rows]), torch.tensor([r["active_mask"] for r in rows]), settings["teacher_temperature"])
        write_jsonl(directory / "supervision.jsonl", [{"pair_id": row["pair_id"], "label": row["label"],
                                                       "teacher": target.tolist(), "covered": bool(flag)}
                                                      for row, target, flag in zip(rows, targets, covered)])
        write_json(directory / "coverage.json", {"covered": int(covered.sum()), "total": len(rows),
                                                "by_gold": {label: {"n": sum(r["label"] == i for r in rows),
                                                                    "covered": sum(bool(flag) for r, flag in zip(rows, covered) if r["label"] == i)} for i, label in enumerate(LABELS)}})

        def eligible(examples):
            if settings["objective"] != "G2":
                return examples
            return [r for r in examples if any(np.argmax(q) == r["label"] for q, enabled in zip(r["expert_probabilities"], r["active_mask"]) if enabled)]

        def loss_fn(current, batch, examples):
            device = batch["input_ids"].device
            return gate_loss(current(batch), torch.tensor([r["expert_probabilities"] for r in examples], device=device),
                             torch.tensor([r["label"] for r in examples], device=device),
                             torch.tensor([r["active_mask"] for r in examples], device=device), settings)

        def validate(current):
            fused, _ = self.gate_probabilities(current, validation)
            return validation_metrics([r["label"] for r in validation], fused)

        def parameter_groups(current):
            return [{"params": current.encoder.parameters(), "lr": settings["learning_rate"]},
                    {"params": current.mlp.parameters(), "lr": settings["mlp_learning_rate"]}]

        selection = fit(model, eligible(train), model.tokenizer, training_settings, self.config["seed"], directory / "selection", loss_fn,
                        validate, lambda current: current.save(directory / "selection" / "checkpoint"), parameter_groups(model))
        release(model)
        seed_everything(self.config["seed"], training_settings["cpu_threads"])
        model = QueryGate.create(self.config["branches"], settings)
        training_settings["epochs"] = selection["best_epoch"]
        fit(model, eligible(rows), model.tokenizer, training_settings, self.config["seed"], directory / "refit", loss_fn,
            None, lambda current: current.save(directory / "checkpoint"), parameter_groups(model))
        write_json(directory / "validation_split.json", {"validation_documents": sorted(validation_docs), "refit_documents": self.train_docs})
        release(model)

    @torch.no_grad()
    def gate_probabilities(self, model, rows, top_k=None):
        model.eval()
        device = next(model.parameters()).device
        probabilities, all_weights = [], []
        for start in range(0, len(rows), self.config["training"]["batch_size"]):
            examples = rows[start:start + self.config["training"]["batch_size"]]
            batch = move_batch(collate(examples, model.tokenizer), device)
            mask = torch.tensor([r["active_mask"] for r in examples], device=device)
            weights = masked_weights(model(batch), mask, model.settings["temperature"], top_k)
            experts = torch.tensor([r["expert_probabilities"] for r in examples], device=device)
            probabilities.extend((weights[:, :, None] * experts).sum(1).cpu().tolist())
            all_weights.extend(weights.cpu().tolist())
        return probabilities, all_weights

    def predict(self, split="dev", fusion="equal", objective=None, top_k=None, drop=None, demo_variant="normal"):
        if split not in ("dev", "test"):
            raise ValueError("Final predictions require dev or test split")
        if top_k is not None and fusion != "learned":
            raise ValueError("top_k requires learned fusion")
        documents = self.dev_docs if split == "dev" else self.test_docs
        if not documents:
            raise ValueError("No documents in requested split")
        records = self.expert_predictions("final", documents, f"{split}_{demo_variant}", demo_variant)
        branches = self.config["branches"]
        modified = copy.deepcopy(records)
        if drop is not None:
            for row in modified:
                row["experts"].pop(drop, None)
        if fusion == "learned" and drop is not None:
            raise ValueError("Drop learned experts requires retraining a gate with that branch removed")
        if fusion == "learned":
            objective = objective or self.config["gate"]["objective"]
            gate = QueryGate.load(self.gate_directory(objective) / "checkpoint").to(device_name(self.config["training"]["device"]))
            rows = self.gate_rows(modified, InputBuilder([gate.tokenizer], "bert_cls", self.config["input"]))
            probabilities, weights = self.gate_probabilities(gate, rows, top_k)
            release(gate)
        else:
            weights, probabilities = [], []
            for row in modified:
                names = [b for b in branches if b in row["experts"]]
                if fusion != "equal":
                    selected = row["features"]["demo_cell"] if fusion == "D" else fusion
                    names = [selected] if selected in names else []
                if not names:
                    raise ValueError("No active expert for prediction")
                w = [1 / len(names) if b in names else 0.0 for b in branches]
                q = np.sum([np.array(row["experts"][b]["probabilities"]) * weight for b, weight in zip(branches, w) if weight], axis=0)
                weights.append(w)
                probabilities.append(q.tolist())
        predictions = [dict(row, probabilities=q, weights=w, branches=branches, label=int(np.argmax(q)),
                            active_mask=[b in row["experts"] for b in branches], status="success")
                       for row, q, w in zip(modified, probabilities, weights)]
        name = f"{split}_{fusion}_{objective or 'none'}_{demo_variant}_top{top_k or 'all'}_drop{drop or 'none'}"
        if fusion == "learned":
            name += "_gate" + digest(self.gate_settings(objective))[:12]
        directory = self.root / "results" / name
        claim_run(directory, {"split": split, "fusion": fusion, "objective": objective, "top_k": top_k,
                              "drop": drop, "demo_variant": demo_variant, "seed": self.config["seed"],
                              "fold": self.fold, "experiment_id": self.config["experiment_id"], "head": self.config["model"]["head"],
                              "gate_settings": self.gate_settings(objective) if fusion == "learned" else None})
        write_jsonl(directory / "predictions.jsonl", predictions)
        write_jsonl(directory / "error_cases.jsonl", [dict(row, gold=self.corpus.by_id[row["pair_id"]]["label"])
                                                     for row in predictions if row["label"] != self.corpus.by_id[row["pair_id"]]["label"]])
        report = evaluate(self.corpus.pairs_for(documents), predictions)
        write_json(directory / "metrics.json", report)
        return directory, report

    def run(self, objectives=("G1", "G2", "G3"), split="dev"):
        self.generate_oof()
        self.train_experts("final")
        self.predict(split, "equal")
        for objective in objectives:
            self.train_gate(objective)
            self.predict(split, "learned", objective)

    def train_graph_expert(self, scope):
        from .graph import train_graph_expert
        return train_graph_expert(self, scope)
