import copy

import pytest

from mode_eci.data import parse_document
from mode_eci.features import RuleRouter, active_mask


def test_query_never_carries_gold(assets):
    _, corpus = assets
    pair = corpus.pairs[0]
    assert corpus.query(pair) == corpus.query(dict(pair, label=(pair["label"] + 1) % 3))
    assert "label" not in corpus.query(pair)
    assert "annotation_audit" not in corpus.query(pair)


def test_cross_cells_and_active_experts(assets):
    _, corpus = assets
    router = RuleRouter("lexical")
    cells = [router(corpus.query(p))["demo_cell"] for p in corpus.pairs[:4]]
    assert cells == ["D1", "D2", "D3", "D4"]
    branches = ["D1", "D2", "D3", "D4", "C", "S", "M"]
    assert active_mask("D2", branches) == [False, True, False, False, True, True, True]


def test_scope_rules_cover_negation_and_unrelated_cues(assets):
    _, corpus = assets
    router = RuleRouter("scope")
    query = copy.deepcopy(corpus.query(corpus.pairs[0]))
    assert router(query)["demo_cell"] == "D1"
    cross = corpus.query(corpus.pairs[2])
    assert router(cross)["demo_cell"] == "D3"
    query = corpus.query(corpus.pairs[1])
    assert router(query)["demo_cell"] == "D2"
    query = copy.deepcopy(corpus.query(corpus.pairs[0]))
    query["sentences"][0]["tokens"] = ["The", "explosion", "did", "not", "cause", "a", "fire", "."]
    query["sentences"][0]["token_ids"] = list(range(101, 109))
    query["e1"]["token_ids"], query["e2"]["token_ids"] = [102], [107]
    query["doc_id"] = "negation_case"
    assert router(query)["demo_cell"] == "D1"


@pytest.mark.parametrize("text,first,second,expected", [
    ("Flooding happened because rain fell .", "Flooding", "rain", "D1"),
    ("Flooding happened because of rain .", "Flooding", "rain", "D1"),
    ("Flooding happened due to rain .", "Flooding", "rain", "D1"),
    ("The explosion led to fire .", "explosion", "fire", "D1"),
    ("Workers approved repairs because rain fell .", "approved", "repairs", "D2"),
])
def test_reason_and_causative_scope(text, first, second, expected):
    words = text.split()
    sentence = {"id": 0, "tokens": words, "token_ids": list(range(len(words)))}
    query = {"pair_id": "scope", "doc_id": text, "topic_id": "test", "sentences": [sentence],
             "e1": {"id": "1", "text": first, "token_ids": [words.index(first)], "sentence_id": 0},
             "e2": {"id": "2", "text": second, "token_ids": [words.index(second)], "sentence_id": 0}}
    assert RuleRouter("scope")(query)["demo_cell"] == expected


def test_split_rejects_document_overlap(assets):
    _, corpus = assets
    modified = copy.deepcopy(corpus)
    modified.splits["folds"][0]["dev"].append(modified.splits["folds"][0]["test"][0])
    with pytest.raises(ValueError, match="leakage"):
        modified.documents_for("0", "train")


def test_esc_direction_uses_extended_labels(tmp_path):
    xml = tmp_path / "doc.xml"
    xml.write_text('''<Document doc_name="1_1ecbplus.xml">
      <token t_id="1" sentence="0">Bridge</token><token t_id="2" sentence="0">fell</token>
      <token t_id="3" sentence="0">after</token><token t_id="4" sentence="0">storm</token>
      <Markables><ACTION_OCCURRENCE m_id="1"><token_anchor t_id="1"/><token_anchor t_id="2"/></ACTION_OCCURRENCE>
      <ACTION_OCCURRENCE m_id="2"><token_anchor t_id="4"/></ACTION_OCCURRENCE>
      <ACTION_OCCURRENCE m_id="3"/></Markables>
      <Relations><PLOT_LINK relType="FALLING_ACTION"><source m_id="1"/><target m_id="2"/></PLOT_LINK></Relations></Document>''', encoding="utf-8")
    extended = tmp_path / "labels.tab"
    extended.write_text("1_2\t4\tPRECONDITION\n", encoding="utf-8")
    doc, pairs = parse_document(xml, extended)
    assert pairs[0]["label"] == 1
    assert len(doc["events"]) == 2
    assert doc["annotation_audit"]["direction_disagreements"] == 1
