import copy

import numpy as np
import pytest

from mode_eci.features import RuleRouter, retrieval_text
from mode_eci.retrieval import DemoIndex, HashEncoder


def make_index(assets):
    config, corpus = assets
    router, encoder = RuleRouter("lexical"), HashEncoder()
    documents = corpus.documents_for("0", "train")
    index = DemoIndex.build(corpus, documents, router, encoder, config["retrieval"], True)
    return index, router, encoder


def test_train_only_four_pools_and_missing_label_policy(assets):
    config, corpus = assets
    index, router, encoder = make_index(assets)
    for pair in corpus.pairs[:4]:
        query = corpus.query(pair)
        features = router(query)
        scores = encoder.encode([retrieval_text(query)])[0] @ index.vectors.T
        plan = index.retrieve(query, features, scores, config["retrieval"], config["seed"])
        assert {r["label"] for r in plan["demos"]} == {0, 1, 2}
        assert all(index.by_id[r["pair_id"]]["features"]["demo_cell"] == features["demo_cell"] for r in plan["demos"])
        assert all(index.by_id[r["pair_id"]]["doc_id"] != query["doc_id"] for r in plan["demos"])
        assert plan == index.retrieve(query, features, scores, config["retrieval"], config["seed"])
    assert set(index.metadata["train_documents"]).isdisjoint(corpus.documents_for("0", "test"))
    index.pools["D1", 2] = np.array([], dtype=int)
    query = corpus.query(corpus.pairs[0])
    plan = index.retrieve(query, router(query), np.zeros(len(index.records)), config["retrieval"], 209)
    assert plan["missing_labels"] == [2] and len(plan["demos"]) == 2


def test_approved_negative_and_independent_structural_recall(assets):
    config, corpus = assets
    index, router, _ = make_index(assets)
    query = corpus.query(corpus.split("0", "dev")[0])
    feature = router(query)
    negatives = index.pools["D1", 0]
    scores = np.zeros(len(index.records))
    scores[negatives[0]], scores[negatives[1]] = 0.99, 0.90
    index.records[negatives[1]]["review_status"] = "approved"
    settings = dict(config["retrieval"], candidates=2)
    plan = index.retrieve(query, feature, scores, settings, 209)
    negative = next(d for d in plan["demos"] if d["label"] == 0)
    assert negative["pair_id"] == index.records[negatives[1]]["pair_id"]
    assert negative["reason"] == "approved_similar_negative"
    positive = index.pools["D1", 1]
    first, outside = positive[0], positive[1]
    for index_value in positive:
        index.records[index_value]["features"] = dict(feature, groups=[], distance_bucket=4, dependency_path=[])
    index.records[outside]["features"] = copy.deepcopy(feature)
    scores[first], scores[outside] = .99, .01
    plan = index.retrieve(query, feature, scores, dict(settings, strategy="R3", candidates=1), 209)
    assert next(d for d in plan["demos"] if d["label"] == 1)["pair_id"] == index.records[outside]["pair_id"]


def test_hash_encoder_cannot_make_research_index(assets):
    config, corpus = assets
    with pytest.raises(ValueError, match="test_only"):
        DemoIndex.build(corpus, corpus.documents_for("0", "train"), RuleRouter("lexical"), HashEncoder(), config["retrieval"], False)
