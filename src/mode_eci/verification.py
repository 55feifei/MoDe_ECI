"""离线微型模型验收，所有数据、模型和指标均标记 test_only。"""

import copy
import itertools
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BartConfig, BartForConditionalGeneration, BertConfig, BertForMaskedLM, BertTokenizerFast, PreTrainedTokenizerFast

from .io import code_fingerprint, read_json, read_jsonl, write_json, write_jsonl


def make_fixture(directory):
    directory = Path(directory)
    documents, pairs = [], []
    patterns = [
        (["The explosion caused a fire ."], [("explosion", 0), ("fire", 0)]),
        (["The blast shattered the windows ."], [("blast", 0), ("shattered", 0)]),
        (["The engine overheated .", "As a result it stopped ."], [("overheated", 0), ("stopped", 1)]),
        (["A pipe burst .", "The basement filled with water ."], [("burst", 0), ("filled", 1)]),
    ]
    for number in range(18):
        doc_id = f"{number // 3 + 1}_fixture_{number}"
        sentences, events = [], []
        offset = 1
        for cell, (texts, targets) in enumerate(patterns):
            base_sentence = len(sentences)
            for text in texts:
                words = text.split()
                sentences.append({"id": len(sentences), "tokens": words, "token_ids": list(range(offset, offset + len(words)))})
                offset += len(words)
            for word, local_sentence in targets:
                sentence = sentences[base_sentence + local_sentence]
                events.append({"id": str(len(events)), "token_ids": [sentence["token_ids"][sentence["tokens"].index(word)]],
                               "sentence_id": sentence["id"], "type": "ACTION_OCCURRENCE", "text": word})
            pairs.append({"pair_id": f"{doc_id}:{2 * cell}:{2 * cell + 1}", "doc_id": doc_id,
                          "event1": 2 * cell, "event2": 2 * cell + 1, "label": number % 3})
        documents.append({"doc_id": doc_id, "topic_id": str(number // 3 + 1), "sentences": sentences, "events": events})
    ids = [doc["doc_id"] for doc in documents]
    split = {"train": ids[:12], "dev": ids[12:15], "test": ids[15:]}
    write_jsonl(directory / "documents.jsonl", documents)
    write_jsonl(directory / "pairs.jsonl", pairs)
    write_json(directory / "splits.json", {"test_only": True, "folds": [split], "pilot": dict(split, test=[]), "fingerprint": "test_only_fixture_v1"})
    return directory


def make_tiny_models(directory):
    directory = Path(directory)
    torch.manual_seed(209)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(
        "the explosion caused a fire blast shattered windows engine overheated as result it stopped pipe burst basement filled with water task identify causal relation between e1 e2 example query context in this event none sentence distance cell d1 d2 d3 d4 . : ; , ->".split()))
    bert = directory / "tiny_bert"
    bert.mkdir(parents=True, exist_ok=True)
    (bert / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(bert / "vocab.txt"))
    tokenizer.save_pretrained(bert)
    model = BertForMaskedLM(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=32, max_position_embeddings=512))
    model.save_pretrained(bert)
    write_json(bert / "source.json", {"test_only": True, "kind": "random_tiny_bert"})
    bart = directory / "tiny_bart"
    bart_words = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + vocabulary[5:]
    raw = Tokenizer(WordLevel(dict(zip(bart_words, range(len(bart_words)))), unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                       pad_token="<pad>", mask_token="<mask>", model_input_names=["input_ids", "attention_mask"])
    tokenizer.save_pretrained(bart)
    model = BartForConditionalGeneration(BartConfig(vocab_size=len(tokenizer), d_model=16, encoder_layers=1, decoder_layers=1,
                                                    encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                                                    decoder_ffn_dim=32, max_position_embeddings=512, pad_token_id=1,
                                                    bos_token_id=0, eos_token_id=2, decoder_start_token_id=2))
    model.save_pretrained(bart)
    write_json(bart / "source.json", {"test_only": True, "kind": "random_tiny_bart"})
    return bert, bart


def fixture_config(base, directory, head="bert_mlm"):
    config = copy.deepcopy(base)
    directory = Path(directory).resolve()
    data = make_fixture(directory / "data")
    bert, bart = make_tiny_models(directory / "models")
    config.update(experiment_id=f"test_only_{head}", test_only=True, data_dir=str(data), output_dir=str(directory / "artifacts"), inner_folds=2)
    config["rules"]["mode"] = "lexical"
    config["model"].update(name=str(bart if head.startswith("bart") else bert), revision=None, head=head)
    config["gate"].update(name=str(bert), revision=None, epochs=1, patience=1, hidden_size=16)
    config["input"].update(max_length=256, demo_max_tokens=55, comparison_tokenizers=[])
    config["training"].update(device="cpu", cpu_threads=1, epochs=1, batch_size=8, accumulation_steps=2, precision="fp32")
    return config


def verify_local(base, directory):
    from .pipeline import Experiment
    summaries = []
    for head in ("bert_mlm", "bart_slots", "bert_cls", "bart_relation"):
        config = fixture_config(base, directory, head)
        if head in ("bert_cls", "bart_relation"):
            config["branches"] = ["S"]
        experiment = Experiment(config, "0")
        if len(config["branches"]) > 1:
            experiment.run()
            # 复用完整阶段的第二次运行验证不会重复训练。
            before = read_json(experiment.root / "final" / "experts" / "S" / "training.json")
            experiment.run()
            assert read_json(experiment.root / "final" / "experts" / "S" / "training.json") == before
        else:
            experiment.train_experts()
        location, report = experiment.predict("test", "equal")
        summaries.append({"head": head, "test_only": report["test_only"], "coverage": report["coverage"], "results": str(location)})
    # 连通的教学候选图用于验收嵌套种子流程，不修改主验收数据。
    config = fixture_config(base, Path(directory) / "graph", "bert_mlm")
    config["branches"] = ["S", "G"]
    config["graph"].update(enabled=True, threshold=0.0)
    documents = list(read_jsonl(Path(config["data_dir"]) / "documents.jsonl"))
    pairs = []
    for number, document in enumerate(documents):
        for first, second in itertools.combinations(range(len(document["events"])), 2):
            pairs.append({"pair_id": f"{document['doc_id']}:{first}:{second}", "doc_id": document["doc_id"],
                          "event1": first, "event2": second, "label": number % 3})
    write_jsonl(Path(config["data_dir"]) / "pairs.jsonl", pairs)
    graph_experiment = Experiment(config, "0")
    graph_experiment.run(objectives=("G3",))
    location, graph_report = graph_experiment.predict("test", "learned", "G3")
    assert graph_report["input_diagnostics"]["expert_availability"]["G"] > 0
    summaries.append({"head": "bert_mlm_nested_graph", "test_only": True, "coverage": graph_report["coverage"], "results": str(location)})
    result = {"status": "test_only", "code_fingerprint": code_fingerprint(),
              "description": "Micro-model software verification; metrics are not research results", "systems": summaries}
    write_json(Path(directory) / "verification.json", result)
    return result
