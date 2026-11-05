"""ESC 数据下载、解析、划分与统一访问。

ESC（EventStoryLine）是本工程使用的英文事件因果识别数据集。每篇文档由
若干句子和事件提及（event mention）组成，两个事件之间可能标注了因果关系。

本模块完成四件事：
1. 从 GitHub 下载固定版本的原始数据；
2. 把 XML 标注和官方评测格式的关系文件解析成统一结构；
3. 生成文档、事件对（pair）和五折划分文件；
4. 提供 ``Corpus`` 类，按“只给上下文、不给答案”的方式查询样本。

标签约定：``NONE`` 表示没有因果关系，``E1_TO_E2`` 表示事件 1 导致事件 2，
``E2_TO_E1`` 表示事件 2 导致事件 1。两个因果方向在“是否存在因果关系”
的评测中会合并。
"""

from __future__ import annotations  # 让类型注解延迟求值（本项目统一保留的写法）

import itertools
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .io import digest, read_json, read_jsonl, write_json, write_jsonl

# 预测标签的顺序必须固定：模型输出的第 0/1/2 个分数分别对应这三个标签。
LABELS = ("NONE", "E1_TO_E2", "E2_TO_E1")

# 上游 ESC 数据仓库。下载时会记录实际使用的 commit，保证数据版本可追溯。
REPO = "tommasoc80/EventStoryLine"
ESC_REVISION = "46edefee5e82e0917b823abe0a18bf8c7770f15c"


def download_esc(destination):
    """下载并解压 ESC v0.9 数据，返回数据来源信息。

    使用 ESC_REVISION 固定的 commit，与 v3 已有数据版本保持一致。

    ``destination/source.json`` 相当于“已下载”的标记文件：如果它存在，
    就直接返回其中记录的信息，不重复下载。只有全部文件都成功解压后，
    才会写入这个标记，因此中途失败不会留下“看似已完成”的状态。

    压缩包只保留三类内容：v0.9 标注 XML、官方评测格式的扩展关系文件、
    许可证。解压前会检查目标路径必须位于 destination 内，防止恶意压缩包
    利用 ``../`` 写到目录之外。
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = destination / "source.json"
    if provenance.exists():
        meta = read_json(provenance)
        if meta["revision"] != ESC_REVISION:
            raise ValueError("ESC source revision differs; use a separate raw directory")
        return meta
    revision = ESC_REVISION
    # 第二步：按固定 SHA 下载源码压缩包。
    archive = destination / "source.zip"
    urllib.request.urlretrieve(f"https://codeload.github.com/{REPO}/zip/{revision}", archive)
    # 只解压 v0.9 相关目录和许可证。
    prefixes = ("annotated_data/v0.9/", "evaluation_format/full_corpus/v0.9/event_mentions_extended/")
    files = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            # GitHub 压缩包顶层通常多一层仓库名目录，split 一次即可去掉它。
            relative = member.filename.split("/", 1)[-1]
            if member.is_dir() or not (relative.startswith(prefixes) or relative == "LICENSE.md"):
                continue
            target = (destination / relative).resolve()
            # 安全检查：解压后的真实路径必须仍在目标目录内。
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
            files += 1
    meta = {"repository": REPO, "revision": revision, "files": files, "version": "v0.9"}
    write_json(provenance, meta)
    return meta


def parse_document(xml_path, extended_path=None):
    """解析一篇 ESC 文档，返回 ``(文档字典, 事件对列表)``。

    参数：
    - ``xml_path``：原始 XML 标注文件；
    - ``extended_path``：官方评测格式的关系文件（制表符分隔）。如果提供且
      文件存在，就以它作为因果关系的唯一权威来源。

    XML 中的 ``PLOT_LINK`` 是原始标注，可能与官方扩展标注存在方向差异。
    本函数不会把两者合并，而是先用原始标注临时计算，再在扩展文件存在时
    整体替换，并记录差异数量供审计。

    事件对标签：``1`` 表示文档中靠前的事件导致靠后的事件，``2`` 表示反向，
    ``0`` 表示没有因果边。事件按第一个 token 的位置排序，所以“靠前/靠后”
    是稳定且可复现的。
    """
    root = ET.parse(xml_path).getroot()
    # doc_name 形如 "1_1ecbplus.xml"，去掉后缀得到文档 ID；下划线前是 topic 编号。
    doc_id = root.attrib["doc_name"].removesuffix(".xml")
    topic_id = doc_id.split("_", 1)[0]
    # 把所有 token 放进字典：t_id -> {文本, 所在句编号}。
    tokens = {int(t.attrib["t_id"]): {"text": t.text or "", "sentence": int(t.attrib["sentence"])}
              for t in root.findall("token")}

    events = []
    # Markables 下可能包含 ACTION_、NEG_ACTION_ 以及共指/抽象节点。
    for mark in root.findall("Markables/*"):
        if not (mark.tag.startswith("ACTION_") or mark.tag.startswith("NEG_ACTION_")):
            continue
        # 一个事件可能对应多个 token_anchor（不连续的事件短语）。
        anchors = sorted(int(t.attrib["t_id"]) for t in mark.findall("token_anchor"))
        if not anchors:  # 没有 token 的是抽象/共指节点，不是事件提及
            continue
        sentence_ids = {tokens[t]["sentence"] for t in anchors}
        if len(sentence_ids) != 1:
            raise ValueError(f"Event spanning multiple sentences: {doc_id}/{anchors}")
        events.append({"id": mark.attrib["m_id"], "token_ids": anchors,
                       "sentence_id": next(iter(sentence_ids)), "type": mark.tag,
                       "text": " ".join(tokens[t]["text"] for t in anchors)})

    # 按第一个 token 的位置排序，并检查同一 token 区间是否重复标注。
    events.sort(key=lambda e: (e["token_ids"][0], e["token_ids"], e["id"]))
    by_id = {e["id"]: i for i, e in enumerate(events)}
    by_span = {}
    for i, event in enumerate(events):
        span = tuple(event["token_ids"])
        if span in by_span:
            raise ValueError(f"Duplicate event span: {doc_id}/{span}")
        by_span[span] = i

    # relations 的键是两个事件下标组成的有序元组 (较小, 较大)，
    # 值 1 表示较小下标是原因，值 2 表示较大下标是原因。
    relations = {}

    def add(source, target, kind):
        """把一条带类型的原始关系转换成 (事件对 -> 方向标签)。"""
        if kind not in ("PRECONDITION", "FALLING_ACTION"):
            # 空值、null 表示“没有关系”，直接忽略；其他未知类型说明数据异常。
            if kind in ("", "NONE", "null", None):
                return
            raise ValueError(f"Unknown relation {kind} in {doc_id}")
        if source == target:
            return
        # ESC 约定：PRECONDITION 的 source 是原因；FALLING_ACTION 的 target 是原因。
        cause, effect = (source, target) if kind == "PRECONDITION" else (target, source)
        key = tuple(sorted((source, target)))
        label = 1 if cause == key[0] else 2
        # 同一对事件出现互相矛盾的方向时拒绝继续，而不是随便选一个。
        if key in relations and relations[key] != label:
            raise ValueError(f"Conflicting directions: {doc_id}/{key}")
        relations[key] = label

    # 先读取 XML 自带的关系。只有两端都是有效事件时才处理。
    for link in root.findall("Relations/PLOT_LINK"):
        source, target = link.find("source"), link.find("target")
        if source is None or target is None:
            raise ValueError(f"Missing relation endpoint in {doc_id}")
        a, b = source.attrib["m_id"], target.attrib["m_id"]
        if a in by_id and b in by_id:
            add(by_id[a], by_id[b], link.get("relType"))
    raw_relations = dict(relations)
    raw_count = len(raw_relations)

    if extended_path and Path(extended_path).exists():
        # 官方 evaluation-format 标注是权威来源，可能修正原始 XML 的方向。
        # 因此清空旧结果重新读取，绝不把两套标注简单合并。
        relations = {}
        for line in Path(extended_path).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            # 每行形如 "1_2<TAB>4<TAB>PRECONDITION"，前两列是事件 token 区间。
            a, b, kind = line.split()
            sa, sb = tuple(map(int, a.split("_"))), tuple(map(int, b.split("_")))
            if sa not in by_span or sb not in by_span:
                raise ValueError(f"Unmatched extended event span in {doc_id}: {a}, {b}")
            add(by_span[sa], by_span[sb], kind)
    elif raw_count:
        # 原始 XML 有因果关系，却没有对应的扩展文件，说明数据不完整，直接报错。
        raise ValueError(f"Missing extended relation file for labelled document {doc_id}")

    # 重新按句子编号组织 token，供后续构造查询文本。
    sentences = []
    for sid in sorted({t["sentence"] for t in tokens.values()}):
        tids = [t for t in sorted(tokens) if tokens[t]["sentence"] == sid]
        sentences.append({"id": sid, "token_ids": tids, "tokens": [tokens[t]["text"] for t in tids]})

    # 审计信息：记录原始标注与官方扩展标注之间的差异，便于排查数据问题。
    audit = {"raw_relations": raw_count, "evaluation_relations": len(relations),
             "direction_disagreements": sum(k in relations and relations[k] != v for k, v in raw_relations.items()),
             "raw_edges_absent_from_evaluation": sum(k not in relations for k in raw_relations)}
    doc = {"doc_id": doc_id, "topic_id": topic_id, "sentences": sentences, "events": events, "annotation_audit": audit}

    # 为文档中所有事件两两组合生成候选对；没有标注关系的组合标签为 0（NONE）。
    pairs = [{"pair_id": f"{doc_id}:{events[a]['id']}:{events[b]['id']}", "doc_id": doc_id,
              "event1": a, "event2": b, "label": relations.get((a, b), 0)}
             for a, b in itertools.combinations(range(len(events)), 2)]
    return doc, pairs


def prepare_esc(raw_dir, output_dir, seed=209):
    """把下载好的原始 ESC 数据转换成工程统一使用的 JSONL 和划分清单。

    划分协议：
    - topic 37、41 固定作为开发集（dev）；
    - 其余文档作为训练池，用固定种子做五折交叉划分；
    - Pilot 实验使用训练池全部文档训练、开发集验证、测试集为空。

    输出文件：
    - ``documents.jsonl``：每行一篇文档；
    - ``pairs.jsonl``：每行一个候选事件对；
    - ``splits.json``：五折、Pilot 和整体数据指纹；
    - ``statistics.json``：数据规模统计。
    """
    from sklearn.model_selection import KFold  # 延迟导入，避免无关命令加载 sklearn
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    source = raw_dir / "annotated_data/v0.9"
    # 文件形如 "1/1_1ecbplus.xml.xml"；先按 topic 数字、再按文件名排序，保证顺序稳定。
    xmls = sorted(source.glob("*/*.xml.xml"), key=lambda p: (int(p.parent.name), p.name))
    if not xmls:
        raise ValueError(f"No ESC v0.9 XML files under {source}")

    docs, pairs = [], []
    for xml in xmls:
        name = xml.name.removesuffix(".xml.xml")
        # 与 XML 同 topic、同文档名的官方扩展关系文件。
        ext = raw_dir / "evaluation_format/full_corpus/v0.9/event_mentions_extended" / xml.parent.name / f"{name}.tab"
        doc, ps = parse_document(xml, ext)
        docs.append(doc)
        pairs.extend(ps)

    if len({d["doc_id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate document identifiers")

    # 固定开发集 topic，其余文档进入五折划分池。
    dev = [d["doc_id"] for d in docs if d["topic_id"] in ("37", "41")]
    pool = [d["doc_id"] for d in docs if d["topic_id"] not in ("37", "41")]
    if not dev or len(pool) < 5:
        raise ValueError("ESC development topics / training documents missing")

    folds = []
    # KFold 会返回每折的训练/测试下标；shuffle=True 与固定 seed 保证可复现。
    for train, test in KFold(5, shuffle=True, random_state=seed).split(pool):
        folds.append({"train": [pool[i] for i in train], "test": [pool[i] for i in test], "dev": dev})

    manifest = {"seed": seed, "protocol": "ilif_document_kfold", "document_order": [d["doc_id"] for d in docs],
                "folds": folds, "pilot": {"train": pool, "dev": dev, "test": []}}
    # 指纹在写入 fingerprint 字段之前计算，避免“指纹包含自己”的循环依赖。
    manifest["fingerprint"] = digest({"documents": docs, "pairs": pairs, "manifest": manifest})

    write_jsonl(output_dir / "documents.jsonl", docs)
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    write_json(output_dir / "splits.json", manifest)
    stats = {"documents": len(docs), "events": sum(len(d["events"]) for d in docs), "pairs": len(pairs),
             "labels": {LABELS[i]: sum(p["label"] == i for p in pairs) for i in range(3)},
             "dev_documents": len(dev), "fingerprint": manifest["fingerprint"],
             "annotation_audit": {key: sum(d["annotation_audit"][key] for d in docs) for key in docs[0]["annotation_audit"]}}
    write_json(output_dir / "statistics.json", stats)
    return stats


class Corpus:
    """统一的数据访问入口：加载文档、事件对和划分，并提供查询接口。

    构造时一次性读入全部数据到内存。``documents`` 用文档 ID 建索引，
    ``by_id`` 用 pair_id 建索引，方便按 ID 快速查找。
    """

    def __init__(self, directory):
        """加载目录中的 documents.jsonl、pairs.jsonl 和 splits.json。"""
        self.directory = Path(directory)
        documents = list(read_jsonl(self.directory / "documents.jsonl"))
        self.documents = {d["doc_id"]: d for d in documents}
        if len(self.documents) != len(documents):
            raise ValueError("Duplicate document IDs")
        self.pairs = list(read_jsonl(self.directory / "pairs.jsonl"))
        self.by_id = {p["pair_id"]: p for p in self.pairs}
        if len(self.by_id) != len(self.pairs):
            raise ValueError("Duplicate pair IDs")
        if any(p["label"] not in (0, 1, 2) for p in self.pairs):
            raise ValueError("Unknown causal label")
        self.splits = read_json(self.directory / "splits.json")
        self.fingerprint = digest({"documents": list(self.documents.values()), "pairs": self.pairs,
                                   "splits": self.splits})
        self.test_only = self.splits.get("test_only", False)

    def documents_for(self, fold, split):
        manifest = self.splits["pilot"] if str(fold) == "pilot" else self.splits["folds"][int(fold)]
        groups = [set(manifest[name]) for name in ("train", "dev", "test")]
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Document split leakage")
        return manifest[split]

    def pairs_for(self, documents):
        ids = set(documents)
        return [p for p in self.pairs if p["doc_id"] in ids]

    def split(self, fold="pilot", split="train"):
        """返回指定折、指定子集（train/dev/test）中的全部事件对。

        ``fold="pilot"`` 使用 Pilot 划分；``fold`` 为 ``"0"`` 到 ``"4"`` 时
        使用对应的五折。每次调用都会检查训练集与开发/测试集是否重叠，
        防止数据泄漏导致指标虚高。
        """
        return self.pairs_for(self.documents_for(fold, split))

    def query(self, pair):
        """把一条事件对转换成推理查询，故意去掉答案和仅用于标注的字段。

        返回值只包含 pair_id、文档 ID、句子、事件 1 和事件 2。注意这里
        **不会**返回 ``pair["label"]``，也不会返回文档的 ``annotation_audit``，
        从而在数据入口处就避免 gold 泄漏。训练时需要的标签由调用方另行
        从 ``pair["label"]`` 读取，不会进入 LLM 查询。
        """
        doc = self.documents[pair["doc_id"]]
        return {"pair_id": pair["pair_id"], "doc_id": doc["doc_id"], "topic_id": doc["topic_id"], "sentences": doc["sentences"],
                "e1": doc["events"][pair["event1"]], "e2": doc["events"][pair["event2"]]}
