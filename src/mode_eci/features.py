"""无 gold 的规则分库和句法特征；词表命中不代表因果标签。"""

import re
from collections import deque

RULE_VERSION = "target_scope_v1"
BRANCHES = ("D1", "D2", "D3", "D4", "C", "S", "M", "G")
CONNECTIVES = {
    "reason": ("because", "because of", "due to", "owing to"),
    "result": ("as a result", "therefore", "thus", "hence", "consequently"),
    "trigger": ("cause", "causes", "caused", "causing", "lead to", "leads to", "led to", "leading to",
                "result in", "results in", "resulted in", "resulting in", "bring about", "brings about", "brought about"),
    "ambiguous": ("as", "since", "so"),
}
PATTERNS = [(term, group, re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, term.split())) + r"(?!\w)", re.I))
            for group, terms in CONNECTIVES.items() for term in terms]


def event_sentences(query):
    ids = {query["e1"]["sentence_id"], query["e2"]["sentence_id"]}
    return [s for s in query["sentences"] if s["id"] in ids]


def sentence_distance(query):
    order = {s["id"]: i for i, s in enumerate(query["sentences"])}
    return abs(order[query["e1"]["sentence_id"]] - order[query["e2"]["sentence_id"]])


def marked_tokens(sentence, query):
    """逐 token 标记，保留不连续和相互重叠的事件 span。"""
    spans = [set(query[e]["token_ids"]) for e in ("e1", "e2")]
    output = []
    for word, tid in zip(sentence["tokens"], sentence["token_ids"]):
        if tid in spans[1]:
            word = f"<E2> {word} </E2>"
        if tid in spans[0]:
            word = f"<E1> {word} </E1>"
        output.append(word)
    return output


def pair_text(query):
    return "\n".join(" ".join(marked_tokens(s, query)) for s in event_sentences(query))


def retrieval_text(query):
    # 标签不参与语义编码，事件表面形式置于开头以降低长句截断影响。
    context = " ".join(" ".join(s["tokens"]) for s in event_sentences(query))
    return f"Event1: {query['e1']['text']}; Event2: {query['e2']['text']}; Context: {context}"


def active_mask(cell, branches, graph_available=False):
    names = {cell, "C", "S", "M"}
    if graph_available:
        names.add("G")
    return [b in names for b in branches]


class RuleRouter:
    """scope 使用依存范围；lexical 复现 v3 的目标句词表代理。

    歧义词在主规则中不单独触发。解析成功但不能连接两目标时记录
    unresolved，分到无线索库；这不是把该样本判为 NONE。
    """

    def __init__(self, mode="scope", spacy_model="en_core_web_sm"):
        self.mode = mode
        self.cache = {}
        self.parser_version = None
        if mode == "scope":
            import spacy
            from spacy.tokens import Doc
            self.nlp = spacy.load(spacy_model, disable=["ner"])
            self.Doc = Doc
            self.parser_version = {"spacy": spacy.__version__, "model": spacy_model,
                                   "version": self.nlp.meta["version"]}

    def parse(self, query, sentence):
        key = (query["doc_id"], sentence["id"])
        if key not in self.cache:
            doc = self.Doc(self.nlp.vocab, words=sentence["tokens"], spaces=[True] * len(sentence["tokens"]))
            self.cache[key] = self.nlp(doc)
        return self.cache[key]

    @staticmethod
    def event_indices(sentence, event):
        ids = set(event["token_ids"])
        return {i for i, tid in enumerate(sentence["token_ids"]) if tid in ids}

    @staticmethod
    def connects(a, b, first, second):
        return bool((a & first and b & second) or (b & first and a & second))

    def scoped_matches(self, query, sentence, matches):
        doc = self.parse(query, sentence)
        a = self.event_indices(sentence, query["e1"])
        b = self.event_indices(sentence, query["e2"])
        distance = sentence_distance(query)
        accepted = []
        for term, group, match in matches:
            span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if span is None or group == "ambiguous":
                continue
            root = span.root
            if group == "trigger":
                # 因果谓词的主语与宾语/补语须分别覆盖目标；否定仍是显式线索。
                subject = {t.i for child in root.children if child.dep_ in ("nsubj", "nsubjpass", "csubj") for t in child.subtree}
                complement = {t.i for child in root.children if child.dep_ in ("dobj", "obj", "attr", "xcomp", "ccomp", "prep", "agent") for t in child.subtree}
                if self.connects(a, b, subject, complement):
                    accepted.append((term, group))
                elif distance == 1 and (a or b) and any(doc[i].lower_ in ("this", "that", "it") for i in subject):
                    # 仅允许后句指示主语回指紧邻前句，不任意跨句扩展。
                    positions = {s["id"]: i for i, s in enumerate(query["sentences"])}
                    other = query["e2"] if a else query["e1"]
                    if positions[sentence["id"]] == positions[other["sentence_id"]] + 1 and (a | b) & complement:
                        accepted.append((term, group))
            elif group == "reason":
                anchor = root.head if root.dep_ == "mark" else root
                if term in ("due to", "owing to"):
                    prepositions = [token for token in span if token.lower_ == "to"]
                    if prepositions:
                        anchor = prepositions[0]
                cause = {t.i for t in anchor.subtree}
                governor = anchor.head
                cue_indices = {token.i for token in span}
                while governor.i in cue_indices and governor.head.i != governor.i:
                    governor = governor.head
                effect = {t.i for t in governor.subtree} - cause
                if a and b and self.connects(a, b, cause, effect) and anchor.dep_ in ("advcl", "prep", "pcomp", "amod", "acl"):
                    accepted.append((term, group))
            elif group == "result":
                positions = {s["id"]: i for i, s in enumerate(query["sentences"])}
                if distance == 1 and (a or b) and span.start <= 2:
                    other = query["e2"] if a else query["e1"]
                    if positions[sentence["id"]] == positions[other["sentence_id"]] + 1:
                        accepted.append((term, group))
                elif a and b:
                    clause = {t.i for t in root.head.subtree}
                    if self.connects(a, b, clause, set(range(len(doc))) - clause):
                        accepted.append((term, group))
        return accepted

    def dependency_path(self, query):
        if self.mode != "scope" or sentence_distance(query):
            return []
        sentence = event_sentences(query)[0]
        doc = self.parse(query, sentence)
        heads = []
        for event in (query["e1"], query["e2"]):
            span = self.event_indices(sentence, event)
            heads.append(next((i for i in sorted(span) if doc[i].head.i not in span or doc[i].head.i == i), min(span)))
        adjacency = {i: [] for i in range(len(doc))}
        for token in doc:
            if token.i != token.head.i:
                adjacency[token.i].append((token.head.i, token.dep_))
                adjacency[token.head.i].append((token.i, token.dep_))
        queue = deque([(heads[0], [])])
        seen = {heads[0]}
        while queue:
            node, path = queue.popleft()
            if node == heads[1]:
                return path
            for neighbor, relation in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [relation]))
        return []

    def __call__(self, query):
        found, accepted = [], []
        for sentence in event_sentences(query):
            text = " ".join(sentence["tokens"])
            matches = [(term, group, m) for term, group, pattern in PATTERNS for m in pattern.finditer(text)]
            found.extend((term, group) for term, group, _ in matches)
            accepted.extend(self.scoped_matches(query, sentence, matches) if self.mode == "scope"
                            else [(term, group) for term, group, _ in matches])
        distance = sentence_distance(query)
        cell = ("D1" if accepted else "D2") if distance == 0 else ("D3" if accepted else "D4")
        return {"demo_cell": cell, "sentence_distance": distance, "distance_bucket": min(distance, 4),
                "connectives": sorted({t for t, _ in accepted}), "groups": sorted({g for _, g in accepted}),
                "lexical_hits": sorted({t for t, _ in found}),
                "unresolved": bool(found and not accepted),
                "ambiguous_hits": sorted({t for t, g in found if g == "ambiguous"}),
                "dependency_path": self.dependency_path(query), "rule_version": RULE_VERSION, "mode": self.mode}
