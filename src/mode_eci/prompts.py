"""确定性证据裁剪和输入模板，训练与推理共用。"""

from .data import LABELS
from .features import event_sentences, marked_tokens

EVENT_TOKENS = ["<E1>", "</E1>", "<E2>", "</E2>"]
RELATION_TOKENS = ["[REL_NONE]", "[REL_12]", "[REL_21]"]
ROLE_TOKENS = ["<None>", "<SLOT>", "<QUERY>"]
STATEMENTS = ["<None> causes <None>.", "<E1> causes <E2>.", "<E2> causes <E1>."]


def register_tokens(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": EVENT_TOKENS + RELATION_TOKENS + ROLE_TOKENS})


def context_text(query, selected):
    lines = []
    for sentence in query["sentences"]:
        positions = selected.get(sentence["id"], set())
        if not positions:
            continue
        words = marked_tokens(sentence, query)
        parts, previous = [], -1
        for position in sorted(positions):
            if position > previous + 1:
                parts.append("...")
            parts.append(words[position])
            previous = position
        if previous < len(words) - 1:
            parts.append("...")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def crop_context(query, sentence_ids, fits):
    """优先保留全部事件 token，再按目标句、间隔句、邻句扩展。

    二分选择确定性优先序的前缀，以免对长文档逐词反复分词。检查的是
    完整模板长度，因而示例和任务说明也计入同一输入预算。
    """
    mandatory = set(query["e1"]["token_ids"]) | set(query["e2"]["token_ids"])
    target_sids = {s["id"] for s in event_sentences(query)}
    positions = {s["id"]: i for i, s in enumerate(query["sentences"])}
    bounds = sorted(positions[s] for s in target_sids)
    selected, candidates = {}, []
    for sentence in query["sentences"]:
        sid = sentence["id"]
        if sid not in sentence_ids:
            continue
        selected[sid] = {i for i, tid in enumerate(sentence["token_ids"]) if tid in mandatory}
        anchors = sorted(selected[sid])
        priority = 0 if sid in target_sids else (1 if bounds[0] < positions[sid] < bounds[-1] else 2)
        for i in range(len(sentence["tokens"])):
            if i not in selected[sid]:
                distance = min(abs(i - anchor) for anchor in anchors) if anchors else min(abs(positions[sid] - x) for x in bounds)
                candidates.append((priority, distance, positions[sid], i, sid))
    candidates.sort()

    def version(count):
        keep = {sid: set(values) for sid, values in selected.items()}
        for _, _, _, index, sid in candidates[:count]:
            keep[sid].add(index)
        return context_text(query, keep), keep

    minimal, _ = version(0)
    if not fits(minimal):
        raise ValueError(f"Mandatory event spans and template exceed input budget: {query['pair_id']}")
    full, keep = version(len(candidates))
    count = len(candidates)
    if not fits(full):
        low, high = 0, len(candidates)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(version(middle)[0]):
                low = middle
            else:
                high = middle - 1
        count = low
        full, keep = version(count)
    retained = sum(len(p) for p in keep.values())
    original = sum(len(s["tokens"]) for s in query["sentences"] if s["id"] in sentence_ids)
    return full, {"retained_sentence_ids": [s["id"] for s in query["sentences"] if keep.get(s["id"])],
                  "original_words": original, "retained_words": retained, "truncated": count < len(candidates)}


class InputBuilder:
    def __init__(self, tokenizers, head, settings):
        self.tokenizers = tokenizers
        self.head = head
        self.settings = settings
        self.demo_cache = {}

    def fits(self, text, maximum):
        # 共同证据预算按较长的陈述式 demo 计数；BERT/BART 不因输出标签
        # 长短不同而保留不同的原文片段。两个 tokenizer 也须共同满足上限。
        canonical = text
        for relation, statement in zip(RELATION_TOKENS, STATEMENTS):
            canonical = canonical.replace(relation, statement)
        if self.tokenizers[0].mask_token is not None:
            canonical = canonical.replace(self.tokenizers[0].mask_token, "<SLOT>")
        for tokenizer in self.tokenizers:
            if len(tokenizer.encode(canonical)) > maximum:
                return False
        return True

    def relation(self, label):
        return STATEMENTS[label] if self.head == "bart_slots" else RELATION_TOKENS[label]

    def demo(self, record, position, label=None):
        label = record["label"] if label is None else label
        key = (record["pair_id"], position, label)
        if key not in self.demo_cache:
            prefix = f"Example {position}: "
            suffix = f"\nRelation: {self.relation(label)}\n"
            query = record["query"]
            text, coverage = crop_context(query, {s["id"] for s in query["sentences"]},
                                          lambda body: self.fits(prefix + body + suffix, self.settings["demo_max_tokens"]))
            self.demo_cache[key] = (prefix + text + suffix, coverage)
        return self.demo_cache[key]

    def build(self, query, branch, features, demos=(), graph_text="", demo_labels=None):
        tokenizer = self.tokenizers[0]
        instruction = "Task: Identify the causal relation between E1 and E2.\n"
        demo_sections, demo_coverage = [], []
        for i, record in enumerate(demos):
            label = None if demo_labels is None else demo_labels[i]
            section, coverage = self.demo(record, i + 1, label)
            demo_sections.append(section)
            demo_coverage.append(coverage)
        prefix = instruction + "".join(demo_sections) + "Query: <QUERY> "
        if branch == "gate":
            suffix = f"\nCell: {features['demo_cell']}; Sentence distance: {features['sentence_distance']}"
        elif self.head == "bert_mlm":
            suffix = f"\nRelation: {tokenizer.mask_token}"
        else:
            suffix = "\nRelation: <SLOT>"
        graph_lines = graph_text.splitlines()
        original_graph_edges = len(graph_lines)
        if branch == "M":
            body = f"E1: <E1> {query['e1']['text']} </E1>; E2: <E2> {query['e2']['text']} </E2>"
            coverage = {"retained_sentence_ids": [], "original_words": 0, "retained_words": 0, "truncated": False}
            if not self.fits(prefix + body + suffix, self.settings["max_length"]):
                raise ValueError(f"Event mentions exceed input budget: {query['pair_id']}")
        else:
            sentences = query["sentences"] if branch == "C" else event_sentences(query)
            if graph_lines:
                # 图是额外证据，不能占掉事件本身的预算；顺序由图距离和置信度确定。
                required = set(query["e1"]["token_ids"]) | set(query["e2"]["token_ids"])
                minimum = context_text(query, {s["id"]: {i for i, tid in enumerate(s["token_ids"]) if tid in required} for s in sentences})
                relation_suffix = suffix
                while graph_lines:
                    suffix = "\nPredicted graph:\n" + "\n".join(graph_lines) + relation_suffix
                    if self.fits(prefix + minimum + suffix, self.settings["max_length"]):
                        break
                    graph_lines.pop()
                if not graph_lines:
                    suffix = relation_suffix
            body, coverage = crop_context(query, {s["id"] for s in sentences},
                                          lambda text: self.fits(prefix + text + suffix, self.settings["max_length"]))
        text = prefix + body + suffix
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        if len(encoded["input_ids"]) > self.settings["max_length"]:
            raise ValueError(f"Encoded input exceeds common budget: {query['pair_id']}")
        # B0 使用事件文本的均值表示。嵌套标记和不连续 span 均由深度计数处理。
        for number in (1, 2):
            start = tokenizer.convert_tokens_to_ids(f"<E{number}>")
            end = tokenizer.convert_tokens_to_ids(f"</E{number}>")
            depth, mask = 0, []
            query_start = encoded["input_ids"].index(tokenizer.convert_tokens_to_ids("<QUERY>")) + 1
            special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
            for position, token in enumerate(encoded["input_ids"]):
                if position >= query_start and token == start:
                    depth += 1
                mask.append(int(position >= query_start and depth > 0 and token not in special))
                if position >= query_start and token == end:
                    depth -= 1
            encoded[f"event{number}_mask"] = mask
        if self.head == "bert_mlm" and branch != "gate":
            matches = [i for i, token in enumerate(encoded["input_ids"]) if token == tokenizer.mask_token_id]
            if len(matches) != 1:
                raise ValueError("Exactly one query MASK required")
            encoded["mask_position"] = matches[0]
        coverage.update({"token_counts": [len(t.encode(text)) for t in self.tokenizers], "demos": demo_coverage,
                         "graph_edges_original": original_graph_edges, "graph_edges_retained": len(graph_lines)})
        return {"encoded": encoded, "text": text, "coverage": coverage}
