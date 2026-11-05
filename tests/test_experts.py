import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mode_eci.experts import Expert, collate, paper_decode
from mode_eci.features import RuleRouter
from mode_eci.prompts import InputBuilder, RELATION_TOKENS


@pytest.mark.parametrize("head", ["bert_mlm", "bert_cls", "bart_relation", "bart_slots"])
def test_gradients_fixed_decoder_and_checkpoint_roundtrip(assets, tmp_path, head):
    config, corpus = assets
    settings = dict(config["model"], head=head)
    if head.startswith("bart"):
        settings["name"] = str(Path(config["model"]["name"]).parent / "tiny_bart")
    torch.set_num_threads(1)
    model = Expert.create(settings)
    builder = InputBuilder([model.tokenizer], head, config["input"])
    query = corpus.query(corpus.pairs[0])
    row = builder.build(query, "S", RuleRouter("lexical")(query))
    batch = collate([row, row], model.tokenizer)
    assert batch["event1_mask"].sum() > 0 and batch["event2_mask"].sum() > 0
    model.eval()
    first = model(batch, torch.tensor([0, 1]))
    second = model(batch, torch.tensor([2, 0]))
    assert torch.allclose(first["probabilities"], second["probabilities"])
    assert torch.isfinite(first["loss"])
    first["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if head.startswith("bart"):
        assert model.tokenizer.convert_tokens_to_ids("<E1>") not in model.fixed_decoder_ids
        assert model.tokenizer.convert_tokens_to_ids("<E2>") not in model.fixed_decoder_ids
        assert model.tokenizer.convert_tokens_to_ids("<None>") not in model.fixed_decoder_ids
        template = model.tokenizer.encode("In this context, <SLOT> causes <SLOT>." if head == "bart_slots" else "Relation: <SLOT>.")
        assert model.fixed_decoder_ids.tolist() == [model.backbone.config.decoder_start_token_id] + template[:-1]
    assert len(set(model.tokenizer.convert_tokens_to_ids(RELATION_TOKENS))) == 3
    model.save(tmp_path / head)
    loaded = Expert.load(tmp_path / head).eval()
    assert torch.allclose(first["probabilities"], loaded(batch)["probabilities"], atol=1e-6)


def test_bart_combination_normalization_and_paper_decoding():
    slots = torch.tensor([[[.15, .70, .10], [.12, .08, .75]]])
    scores = torch.stack([slots[:, 0, 0] * slots[:, 1, 0], slots[:, 0, 1] * slots[:, 1, 2], slots[:, 0, 2] * slots[:, 1, 1]], dim=-1)
    assert np.allclose((scores / scores.sum(-1, keepdim=True))[0], [.0326679, .9528131, .0145191], atol=1e-6)
    assert paper_decode(slots, 0.3).tolist() == [1]


def test_cropping_preserves_discontinuous_and_overlapping_events(assets):
    config, corpus = assets
    model = Expert.create(config["model"])
    query = copy.deepcopy(corpus.query(corpus.pairs[0]))
    sentence = query["sentences"][0]
    sentence["tokens"] = ["context"] * 1000
    sentence["token_ids"] = list(range(1, 1001))
    query["e1"]["token_ids"], query["e2"]["token_ids"] = [1, 500], [500, 1000]
    builder = InputBuilder([model.tokenizer], "bert_mlm", dict(config["input"], max_length=64))
    row = builder.build(query, "C", RuleRouter("lexical")(query))
    assert len(row["encoded"]["input_ids"]) <= 64
    assert row["text"].count("<E1>") == 2 and row["text"].count("<E2>") == 2
    assert sum(row["encoded"]["event1_mask"]) == 2
    assert sum(row["encoded"]["event2_mask"]) == 2


def test_bert_and_bart_keep_identical_evidence_under_common_budget(assets):
    config, corpus = assets
    bert = Expert.create(config["model"])
    bart_settings = dict(config["model"], name=str(Path(config["model"]["name"]).parent / "tiny_bart"), head="bart_slots")
    bart = Expert.create(bart_settings)
    query = copy.deepcopy(corpus.query(corpus.pairs[0]))
    sentence = query["sentences"][0]
    sentence["tokens"], sentence["token_ids"] = ["context"] * 400, list(range(1, 401))
    query["e1"]["token_ids"], query["e2"]["token_ids"] = [1], [400]
    demo_pair = corpus.pairs[4]
    demo = {"pair_id": demo_pair["pair_id"], "query": corpus.query(demo_pair), "label": 1}
    settings = dict(config["input"], max_length=128, demo_max_tokens=45)
    features = RuleRouter("lexical")(query)
    first = InputBuilder([bert.tokenizer, bart.tokenizer], "bert_mlm", settings).build(query, "D1", features, [demo])
    second = InputBuilder([bart.tokenizer, bert.tokenizer], "bart_slots", settings).build(query, "D1", features, [demo])
    assert first["coverage"]["retained_words"] == second["coverage"]["retained_words"]
    assert first["text"].split("<QUERY>")[1].split("\nRelation:")[0] == second["text"].split("<QUERY>")[1].split("\nRelation:")[0]


def test_graph_truncation_keeps_event_tokens(assets):
    config, corpus = assets
    model = Expert.create(config["model"])
    query = corpus.query(corpus.pairs[0])
    builder = InputBuilder([model.tokenizer], "bert_mlm", dict(config["input"], max_length=96))
    graph = "\n".join(f"event_{i} (explosion) -> event_{i+1} (fire), confidence=0.9500" for i in range(12))
    row = builder.build(query, "G", RuleRouter("lexical")(query), graph_text=graph)
    assert len(row["encoded"]["input_ids"]) <= 96
    assert 0 < row["coverage"]["graph_edges_retained"] < row["coverage"]["graph_edges_original"]
    assert sum(row["encoded"]["event1_mask"]) > 0 and sum(row["encoded"]["event2_mask"]) > 0


def test_modified_checkpoint_is_rejected(assets, tmp_path):
    config, _ = assets
    model = Expert.create(config["model"])
    model.save(tmp_path)
    path = tmp_path / "tokenizer" / "tokenizer_config.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        Expert.load(tmp_path)
