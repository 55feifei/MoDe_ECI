import copy
from pathlib import Path

import pytest

from mode_eci.calibration import fit_temperature
from mode_eci.graph import PredictedGraph
from mode_eci.io import digest, write_json, write_jsonl
from mode_eci.pipeline import Experiment, document_groups


def test_document_groups_are_disjoint_and_complete(assets):
    config, corpus = assets
    documents = corpus.documents_for("0", "train")
    groups = document_groups(corpus, documents, 3, config["seed"])
    assert len(set(sum(groups, []))) == len(documents)
    assert sorted(sum(groups, [])) == sorted(documents)
    assert groups == document_groups(corpus, documents, 3, config["seed"])


def test_oof_supervision_rejects_training_overlap(assets, tmp_path):
    config, corpus = assets
    config = copy.deepcopy(config)
    config["output_dir"] = str(tmp_path)
    experiment = Experiment(config, "0")
    records = [{"pair_id": p["pair_id"], "doc_id": p["doc_id"], "source_scope": "bad_source"} for p in corpus.pairs_for(experiment.train_docs)]
    write_jsonl(experiment.root / "oof" / "records.jsonl", records)
    write_json(experiment.root / "oof" / "provenance.json", {"records_fingerprint": digest(records),
               "sources": [{"scope": "bad_source", "train_documents": experiment.train_docs, "held_documents": experiment.train_docs}]})
    with pytest.raises(ValueError, match="leakage"):
        experiment.supervised_oof()


def test_graph_removes_both_query_directions_and_ignores_gold(assets):
    _, original = assets
    corpus = copy.deepcopy(original)
    base = corpus.pairs[0]
    query = corpus.query(base)
    document = base["doc_id"]
    records = [{"pair_id": base["pair_id"], "probabilities": [.01, .98, .01]}]
    for suffix, first, second in [("a", 0, 2), ("b", 2, 1)]:
        identifier = f"{document}:extra:{suffix}"
        corpus.by_id[identifier] = {"pair_id": identifier, "doc_id": document, "event1": first, "event2": second, "label": 0}
        records.append({"pair_id": identifier, "probabilities": [.01, .98, .01]})
    graph = PredictedGraph(corpus, records, {"threshold": .9, "max_edges": 12, "max_hops": 2})
    text = graph.serialize(query)
    assert "event_0 (explosion) -> event_1 (fire)" not in text
    assert text.count("confidence=") == 2
    assert graph.serialize(query, shuffled=True).count("confidence=") == 2
    for pair in corpus.by_id.values():
        pair["label"] = 2
    assert PredictedGraph(corpus, records, graph.settings).serialize(query) == text


def test_temperature_fit_does_not_raise_nll():
    result = fit_temperature([[9., 1., 0.], [1., 9., 0.], [9., 1., 0.], [1., 9., 0.]], [0, 1, 1, 0])
    assert result["temperature"] > 0
    assert result["nll_after"] <= result["nll_before"]
