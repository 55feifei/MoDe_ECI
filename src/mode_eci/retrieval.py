"""四库 MPNet 检索、独立结构召回与可审核难负例。"""

import csv
import hashlib
import random
from pathlib import Path

import numpy as np

from .features import event_sentences, retrieval_text
from .io import digest, file_digest, read_json, read_jsonl, write_json, write_jsonl


class HashEncoder:
    """仅供离线软件验证，所有下游产物必须携带 test_only。"""

    test_only = True
    identity = "test_only_hash_v1"

    def encode(self, texts):
        result = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:2], "little") % 64
                result[row, index] += 1
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


class MPNetEncoder:
    test_only = False

    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer
        self.identity = {"name": settings["name"], "revision": settings["revision"]}
        self.model = SentenceTransformer(settings["name"], revision=settings["revision"], device=settings["device"])
        self.batch_size = settings["batch_size"]

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 100)


def read_reviews(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review pair_id")
    statuses = {r["pair_id"]: r["review_status"] for r in rows}
    if not set(statuses.values()) <= {"approved", "rejected", "ambiguous", "unreviewed", ""}:
        raise ValueError("Unknown review_status")
    return statuses


class DemoIndex:
    def __init__(self, records, vectors, metadata):
        self.records = records
        self.vectors = vectors
        self.metadata = metadata
        self.by_id = {r["pair_id"]: r for r in records}
        self.doc_ids = np.array([r["doc_id"] for r in records])
        self.topic_ids = np.array([r["topic_id"] for r in records])
        self.stable_ranks = np.argsort(np.argsort([r["pair_id"] for r in records]))
        self.pools = {}
        for cell in ("D1", "D2", "D3", "D4", "single"):
            for label in range(3):
                self.pools[cell, label] = np.array([i for i, r in enumerate(records) if r["label"] == label
                                                   and (cell == "single" or r["features"]["demo_cell"] == cell)], dtype=int)

    @classmethod
    def build(cls, corpus, documents, router, encoder, settings, test_only):
        if encoder.test_only and not test_only:
            raise ValueError("HashEncoder is test_only")
        reviews = read_reviews(settings["review_file"])
        records = []
        for pair in corpus.pairs_for(documents):
            query = corpus.query(pair)
            features = router(query)
            # demo 不需要重复保存整篇文档，目标句足够重建 D 视角。
            demo_query = dict(query, sentences=event_sentences(query))
            records.append({"pair_id": pair["pair_id"], "doc_id": pair["doc_id"], "topic_id": query["topic_id"],
                            "query": demo_query, "label": pair["label"], "features": features,
                            "review_status": reviews.get(pair["pair_id"], "unreviewed")})
        if not records:
            raise ValueError("No training pairs in demo index")
        vectors = encoder.encode([retrieval_text(r["query"]) for r in records])
        mentions = sorted({r["query"][event]["text"] for r in records for event in ("e1", "e2")})
        mention_vectors = dict(zip(mentions, encoder.encode(mentions)))
        for record in records:
            query = record["query"]
            record["event_similarity"] = float(mention_vectors[query["e1"]["text"]] @ mention_vectors[query["e2"]["text"]])
        metadata = {"train_documents": sorted(documents), "encoder": encoder.identity,
                    "settings": settings, "parser": router.parser_version, "rule_mode": router.mode,
                    "corpus": corpus.fingerprint, "test_only": test_only,
                    "review_sha256": None if settings["review_file"] is None else file_digest(settings["review_file"])}
        metadata["vectors_sha256"] = hashlib.sha256(vectors.tobytes()).hexdigest()
        metadata["records_sha256"] = digest(records)
        metadata["fingerprint"] = digest(metadata)
        return cls(records, vectors, metadata)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "records.jsonl", self.records)
        np.save(directory / "vectors.npy", self.vectors)
        write_json(directory / "metadata.json", self.metadata)
        with (directory / "negative_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["pair_id", "demo_cell", "event_similarity", "text", "review_status", "reason"])
            writer.writeheader()
            for r in self.records:
                if r["label"] == 0:
                    writer.writerow({"pair_id": r["pair_id"], "demo_cell": r["features"]["demo_cell"],
                                     "event_similarity": r["event_similarity"], "text": retrieval_text(r["query"]),
                                     "review_status": r["review_status"], "reason": ""})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        records = list(read_jsonl(directory / "records.jsonl"))
        vectors = np.load(directory / "vectors.npy")
        metadata = read_json(directory / "metadata.json")
        if digest(records) != metadata["records_sha256"] or hashlib.sha256(vectors.tobytes()).hexdigest() != metadata["vectors_sha256"]:
            raise ValueError("Demo index fingerprint differs")
        return cls(records, vectors, metadata)

    @staticmethod
    def structure_level(query_features, demo_features):
        same_group = bool(set(query_features["groups"]) & set(demo_features["groups"]))
        same_distance = query_features["distance_bucket"] == demo_features["distance_bucket"]
        path = query_features["dependency_path"]
        same_path = bool(path) and path == demo_features["dependency_path"]
        return int(same_group) + int(same_distance) + int(same_path)

    def retrieve(self, query, features, scores, settings, seed):
        strategy = settings["strategy"]
        stable_seed = int(digest([seed, query["pair_id"]])[:16], 16)
        randomizer = random.Random(stable_seed)
        labels = [0, 1, 2] if settings["demo_count"] == 3 else [0, 1 + stable_seed % 2]
        chosen, missing, reasons = [], [], []
        for label in labels:
            cell = "single" if settings["pool"] == "single" else features["demo_cell"]
            pool = self.pools[cell, label]
            pool = pool[self.doc_ids[pool] != query["doc_id"]]
            if len(pool) == 0:
                missing.append(label)
                continue
            if strategy == "R0":
                selected = randomizer.choice(sorted(pool, key=lambda i: self.records[i]["pair_id"]))
                reason = "random_same_label"
            else:
                order = np.lexsort((self.stable_ranks[pool], -scores[pool]))[:settings["candidates"]]
                semantic = pool[order].tolist()
                candidates = semantic
                reason = "semantic_nearest"
                if label == 0 and strategy in ("R2", "R3"):
                    hard = [i for i in semantic if self.records[i]["review_status"] == "approved"
                            and scores[i] >= settings["hard_min_similarity"]]
                    if hard:
                        candidates, reason = hard, "approved_similar_negative"
                    else:
                        reason = "ordinary_negative_no_approved_similar_candidate"
                elif label != 0 and strategy == "R3":
                    # 结构召回扫描正例全池，不能先用语义 top N 裁掉跨主题样本。
                    structural = [i for i in pool if self.records[i]["topic_id"] != query["topic_id"]
                                  and self.structure_level(features, self.records[i]["features"]) > 0]
                    if structural:
                        candidates = list(set(semantic) | set(structural))
                        reason = "semantic_plus_cross_topic_structure"
                if strategy == "R3" and label != 0:
                    selected = min(candidates, key=lambda i: (-self.structure_level(features, self.records[i]["features"]),
                                                             -float(scores[i]), self.records[i]["pair_id"]))
                else:
                    selected = min(candidates, key=lambda i: (-float(scores[i]), self.records[i]["pair_id"]))
            record = self.records[int(selected)]
            chosen.append({"pair_id": record["pair_id"], "similarity": float(scores[selected]),
                           "same_topic": record["topic_id"] == query["topic_id"], "label": label,
                           "structure_level": self.structure_level(features, record["features"]), "reason": reason})
            reasons.append(reason)
        randomizer.shuffle(chosen)
        return {"query_id": query["pair_id"], "demos": chosen, "missing_labels": missing, "reasons": reasons,
                "index_fingerprint": self.metadata["fingerprint"]}

    def retrieve_many(self, queries, features, encoder, settings, seed):
        vectors = encoder.encode([retrieval_text(q) for q in queries])
        rows = []
        # 分块点积避免建立 N_query × N_train 的完整相似度矩阵。
        for start in range(0, len(queries), 128):
            scores = vectors[start:start + 128] @ self.vectors.T
            for offset, score in enumerate(scores):
                index = start + offset
                rows.append(self.retrieve(queries[index], features[index], score, settings, seed))
        return rows
