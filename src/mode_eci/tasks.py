"""任务标签和产物协议；原始语料与检索分层始终使用方向标签。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    relation_tokens: tuple[str, ...]
    statements: tuple[str, ...]

    @property
    def num_labels(self):
        return len(self.labels)

    def label(self, raw_label):
        if raw_label not in (0, 1, 2):
            raise ValueError(f"Unknown causal label: {raw_label}")
        return int(raw_label != 0) if self.name == "existence" else int(raw_label)

    def metadata(self):
        return {"task": self.name, "label_names": list(self.labels)}

    def check(self, record):
        # 旧产物没有 task，严格按方向任务解释，不能按概率维数猜测。
        if record.get("task", "direction") != self.name:
            raise ValueError(f"Task mismatch: expected {self.name}, got {record.get('task', 'direction')}")
        if "label_names" in record and record["label_names"] != list(self.labels):
            raise ValueError("Task label order differs")

    def check_probabilities(self, probabilities):
        if len(probabilities) != self.num_labels:
            raise ValueError(f"Invalid probability dimension for task {self.name}")


TASKS = {
    "direction": Task("direction", ("NONE", "E1_TO_E2", "E2_TO_E1"),
                      ("[REL_NONE]", "[REL_12]", "[REL_21]"),
                      ("<None> causes <None>.", "<E1> causes <E2>.", "<E2> causes <E1>.")),
    "existence": Task("existence", ("NONE", "CAUSAL"), ("[REL_NONE]", "[REL_CAUSAL]"),
                      ("No causal relation exists between <E1> and <E2>.",
                       "A causal relation exists between <E1> and <E2>.")),
}


def get_task(name="direction"):
    if isinstance(name, Task):
        return name
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}")
    return TASKS[name]


def records_task(records, expected=None):
    task = get_task(expected if expected is not None else (records[0].get("task", "direction") if records else "direction"))
    for row in records:
        task.check(row)
        if "probabilities" in row:
            task.check_probabilities(row["probabilities"])
        for expert in row.get("experts", {}).values():
            task.check_probabilities(expert["probabilities"])
    return task
