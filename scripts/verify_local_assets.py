"""只做真实本地资产的加载和少量前向验证，不训练、不计算研究指标。"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mode_eci.data import Corpus
from mode_eci.experts import Expert, collate
from mode_eci.features import RuleRouter, retrieval_text
from mode_eci.io import code_fingerprint, write_json
from mode_eci.prompts import InputBuilder
from mode_eci.retrieval import DemoIndex, MPNetEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--mpnet", required=True)
    parser.add_argument("--output", default="runs/local_assets/verification.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    corpus = Corpus(args.data_dir)
    router = RuleRouter("scope")
    train_docs = corpus.documents_for("0", "train")
    query_pairs = corpus.split("0", "dev")[:3]
    queries = [corpus.query(pair) for pair in query_pairs]
    settings = {"name": args.mpnet, "revision": None, "device": "cpu", "batch_size": 16,
                "strategy": "R2", "pool": "four", "demo_count": 3, "candidates": 50,
                "hard_min_similarity": .5, "review_file": None}
    encoder = MPNetEncoder(settings)
    vectors = encoder.encode([retrieval_text(q) for q in queries])
    assert vectors.shape == (3, 768)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    # 只取三个真实训练文档建小索引，并始终标 test_only。
    index = DemoIndex.build(corpus, train_docs[:3], router, encoder, settings, test_only=True)
    plans = index.retrieve_many(queries, [router(q) for q in queries], encoder, settings, 209)
    assert all(index.by_id[demo["pair_id"]]["doc_id"] != query["doc_id"] for query, plan in zip(queries, plans) for demo in plan["demos"])
    model = Expert.create({"name": args.bert, "revision": None, "head": "bert_mlm", "temperature": 1.,
                           "slot_relation_weight": 0., "gradient_checkpointing": False}).eval()
    builder = InputBuilder([model.tokenizer], model.head, {"max_length": 256, "demo_max_tokens": 60})
    rows = [builder.build(q, "S", router(q)) for q in queries]
    with torch.no_grad():
        result = model(collate(rows, model.tokenizer))
    assert torch.allclose(result["probabilities"].sum(-1), torch.ones(3), atol=1e-6)
    output = {"status": "test_only", "code_fingerprint": code_fingerprint(), "corpus_fingerprint": corpus.fingerprint,
              "documents": len(corpus.documents), "pairs": len(corpus.pairs), "query_count": 3,
              "mpnet_dimensions": 768, "index_pairs": len(index.records), "index_test_only": index.metadata["test_only"],
              "scope_rule_cells": [router(q)["demo_cell"] for q in queries],
              "bert_probability_shape": list(result["probabilities"].shape), "formal_training": False}
    write_json(args.output, output)
    print(output)


if __name__ == "__main__":
    main()
