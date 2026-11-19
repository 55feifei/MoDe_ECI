"""完整覆盖评测、概率诊断、专家互补性与配对文档 bootstrap。"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from .data import LABELS
from .tasks import get_task, records_task


def prf(gold, predicted, directional=False, *, reverse_as_fp_only=False):
    """可选方向口径下，反向仍计 FP，只有预测为 NONE 的正例计 FN。"""
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    positive_gold, positive_pred = gold != 0, predicted != 0
    correct = gold == predicted if directional else positive_gold == positive_pred
    tp = int(np.sum(positive_gold & positive_pred & correct))
    fp, fn = int(positive_pred.sum()) - tp, int(positive_gold.sum()) - tp
    if directional and reverse_as_fp_only:
        fn = int(np.sum(positive_gold & ~positive_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    defined = bool(positive_gold.any())
    f1 = 2 * tp / (2 * tp + fp + fn) if defined else None
    return {"n": len(gold), "gold_positive": int(positive_gold.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "defined": defined}


def validation_metrics(gold, probabilities, task="direction"):
    task = get_task(task)
    result = metrics([task.label(label) for label in gold], probabilities, task=task)
    primary = result[task.name]["f1"]
    return {"selection_f1": primary if primary is not None else 0.0,
            f"{task.name}_f1": primary if primary is not None else 0.0, "nll": result["nll"]}


def reliability(probabilities, outcomes, bins=10):
    probabilities, outcomes = np.asarray(probabilities), np.asarray(outcomes)
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = assignments == index
        rows.append({"lower": index / bins, "upper": (index + 1) / bins, "n": int(selected.sum()),
                     "mean_probability": float(probabilities[selected].mean()) if selected.any() else None,
                     "observed_frequency": float(outcomes[selected].mean()) if selected.any() else None})
    ece = sum(r["n"] * abs(r["mean_probability"] - r["observed_frequency"]) for r in rows if r["n"]) / len(outcomes)
    return {"ece": ece, "bins": rows}


def metrics(gold, probabilities, labels=None, task="direction"):
    task = get_task(task)
    gold, probabilities = np.asarray(gold, dtype=int), np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(gold), task.num_labels) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError(f"Invalid {task.num_labels}-class probabilities")
    if not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    labels = probabilities.argmax(-1) if labels is None else np.asarray(labels)
    if labels.shape != gold.shape or not np.isin(gold, range(task.num_labels)).all() or not np.isin(labels, range(task.num_labels)).all():
        raise ValueError("Invalid task labels")
    result = {"n": len(gold), "existence": prf(gold, labels)}
    if task.name == "direction":
        result.update(direction=prf(gold, labels, directional=True),
                      direction_fp_only=prf(gold, labels, directional=True, reverse_as_fp_only=True))
    if not len(gold):
        if task.name == "existence":
            result.update(nll=None, brier=None, existence_average_precision=None, class_reliability={})
        return result
    onehot = np.eye(task.num_labels)[gold]
    result.update(nll=float(-np.log(np.maximum(probabilities[np.arange(len(gold)), gold], 1e-12)).mean()),
                  brier=float(np.sum((probabilities - onehot) ** 2, axis=1).mean()),
                  existence_average_precision=float(average_precision_score(gold != 0, probabilities[:, 1:].sum(-1))) if np.any(gold) else None,
                  class_reliability={name: reliability(probabilities[:, i], gold == i) for i, name in enumerate(task.labels)})
    return result


def align_predictions(pairs, predictions, task=None):
    expected = {p["pair_id"] for p in pairs}
    by_id = {p["pair_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Predictions missing, duplicated, or outside evaluation manifest")
    specification = records_task(predictions, task)
    if any(row["status"] != "success" or row["label"] not in range(specification.num_labels) for row in predictions):
        raise ValueError("Failed predictions cannot be scored as NONE")
    return [by_id[p["pair_id"]] for p in pairs]


def evaluate(pairs, predictions, task=None):
    task = records_task(predictions, task)
    predictions = align_predictions(pairs, predictions, task)
    gold = np.array([task.label(p["label"]) for p in pairs], dtype=int)
    probabilities = np.array([p["probabilities"] for p in predictions]) if predictions else np.empty((0, task.num_labels))
    labels = np.array([p["label"] for p in predictions])
    groups = defaultdict(list, overall=[], intra=[], cross=[])
    if task.name == "existence":
        groups.update({branch: [] for branch in ("D1", "D2", "D3", "D4")})
    for index, row in enumerate(predictions):
        feature = row["features"]
        for name in ("overall", feature["demo_cell"], "intra" if feature["sentence_distance"] == 0 else "cross",
                     f"distance_{feature['distance_bucket']}", f"label_{task.labels[gold[index]]}"):
            groups[name].append(index)
    report = {name: metrics(gold[indices], probabilities[indices], labels[indices], task) for name, indices in groups.items()}
    report.update(task.metadata())
    report["test_only"] = any(p["test_only"] for p in predictions)
    report["coverage"] = 1.0
    report["label_counts"] = dict(Counter(task.labels[label] for label in gold))
    report["complementarity"] = complementarity(pairs, predictions, task)
    report["input_diagnostics"] = input_diagnostics(predictions, task)
    return report


def input_diagnostics(predictions, task="direction"):
    task = get_task(task)
    truncated, counts, following, missing = Counter(), Counter(), Counter(), Counter()
    missing_strata = Counter()
    for row in predictions:
        for branch, expert in row["experts"].items():
            counts[branch] += 1
            truncated[branch] += expert["coverage"]["truncated"]
            plan = expert["retrieval"]
            if plan is not None:
                for label in plan["missing_labels"]:
                    missing_strata[LABELS[label]] += 1
                present = {task.label(d["label"]) for d in plan["demos"]}
                for label in {task.label(label) for label in plan["missing_labels"]} - present:
                    missing[task.labels[label]] += 1
                displayed = plan.get("displayed_labels", [task.label(d["label"]) for d in plan["demos"]])
                if displayed:
                    following["n"] += 1
                    following["last_demo_label_matches_expert"] += int(np.argmax(expert["probabilities"])) == displayed[-1]
    return {"expert_availability": {branch: count / len(predictions) for branch, count in counts.items()},
            "truncation_rate": {branch: truncated[branch] / count for branch, count in counts.items()},
            "demo_missing_labels": dict(missing), "demo_missing_strata": dict(missing_strata),
            "last_demo_follow_counts": dict(following)}


def complementarity(pairs, predictions, task="direction"):
    task = get_task(task)
    coverage = Counter()
    class_coverage = defaultdict(Counter)
    weights, errors = defaultdict(list), defaultdict(list)
    for pair, row in zip(pairs, predictions):
        gold_label = task.label(pair["label"])
        flags = []
        for branch, expert in row["experts"].items():
            correct = int(np.argmax(expert["probabilities"])) == gold_label
            flags.append(correct)
            errors[branch].append((pair["pair_id"], int(not correct)))
        name = "all_correct" if all(flags) else ("all_wrong" if not any(flags) else "mixed")
        coverage[name] += 1
        class_coverage[task.labels[gold_label]][name] += 1
        for branch, weight in zip(row["branches"], row["weights"]):
            weights[(row["features"]["demo_cell"], branch)].append(weight)
    correlations = {}
    names = sorted(errors)
    for a, first in enumerate(names):
        left = dict(errors[first])
        for second in names[a + 1:]:
            right = dict(errors[second])
            shared = sorted(left.keys() & right.keys())
            x, y = [left[k] for k in shared], [right[k] for k in shared]
            correlations[f"{first}:{second}"] = float(np.corrcoef(x, y)[0, 1]) if len(shared) > 1 and np.std(x) and np.std(y) else None
    return {"counts": dict(coverage), "by_gold": dict(class_coverage), "error_correlations": correlations,
            "mean_weights_by_cell": {f"{cell}:{branch}": float(np.mean(values)) for (cell, branch), values in weights.items()}}


def bootstrap(pairs, first, second, repeats=2000, seed=209, by_topic=False, metric=None):
    first, second = align_predictions(pairs, first), align_predictions(pairs, second)
    first_task, second_task = records_task(first), records_task(second)
    if metric is None:
        if first_task != second_task:
            raise ValueError("Cross-task comparison requires --metric existence")
        metric = first_task.name
    if metric not in ("direction", "existence"):
        raise ValueError(f"Unknown comparison metric: {metric}")
    if metric == "direction" and (first_task.name != "direction" or second_task.name != "direction"):
        raise ValueError("Direction comparison requires direction predictions")
    documents = defaultdict(list)
    for i, pair in enumerate(pairs):
        key = pair["doc_id"].split("_", 1)[0] if by_topic else pair["doc_id"]
        documents[key].append(i)
    counts = []
    for indices in documents.values():
        gold = [pairs[i]["label"] for i in indices]
        row = []
        for predictions in (first, second):
            result = prf(gold, [predictions[i]["label"] for i in indices], metric == "direction")
            row.append([result[k] for k in ("tp", "fp", "fn")])
        counts.append(row)
    counts = np.array(counts).reshape(-1, 2, 3)
    randomizer = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        if not len(counts):
            break
        sample = counts[randomizer.integers(0, len(counts), len(counts))].sum(0)
        denominator = 2 * sample[:, 0] + sample[:, 1] + sample[:, 2]
        if np.all(denominator):
            f1 = 2 * sample[:, 0] / denominator
            differences.append(f1[1] - f1[0])
    return {"metric": metric, "unit": "topic" if by_topic else "document", "repeats": repeats, "valid_repeats": len(differences),
            f"delta_{metric}_f1_interval_95": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
            "seed": seed}
