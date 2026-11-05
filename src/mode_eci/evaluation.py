"""完整覆盖评测、概率诊断、专家互补性与配对文档 bootstrap。"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from .data import LABELS


def prf(gold, predicted, directional=False):
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    positive_gold, positive_pred = gold != 0, predicted != 0
    correct = gold == predicted if directional else positive_gold == positive_pred
    tp = int(np.sum(positive_gold & positive_pred & correct))
    fp, fn = int(positive_pred.sum()) - tp, int(positive_gold.sum()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    defined = bool(positive_gold.any())
    f1 = 2 * tp / (2 * tp + fp + fn) if defined else None
    return {"n": len(gold), "gold_positive": int(positive_gold.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "defined": defined}


def validation_metrics(gold, probabilities):
    gold = np.asarray(gold)
    probabilities = np.asarray(probabilities)
    direction = prf(gold, probabilities.argmax(-1), True)
    nll = float(-np.log(np.maximum(probabilities[np.arange(len(gold)), gold], 1e-12)).mean())
    return {"direction_f1": direction["f1"] if direction["defined"] else 0.0, "nll": nll}


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


def metrics(gold, probabilities, labels=None):
    gold, probabilities = np.asarray(gold), np.asarray(probabilities, dtype=float)
    labels = probabilities.argmax(-1) if labels is None else np.asarray(labels)
    if probabilities.shape != (len(gold), 3) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid three-class probabilities")
    if not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    if not len(gold):
        return {"n": 0, "direction": prf([], [], True), "existence": prf([], [])}
    onehot = np.eye(3)[gold]
    result = {"n": len(gold), "direction": prf(gold, labels, True), "existence": prf(gold, labels),
              "nll": float(-np.log(np.maximum(probabilities[np.arange(len(gold)), gold], 1e-12)).mean()),
              "brier": float(np.sum((probabilities - onehot) ** 2, axis=1).mean()),
              "existence_average_precision": float(average_precision_score(gold != 0, probabilities[:, 1:].sum(-1))) if np.any(gold) else None,
              "class_reliability": {name: reliability(probabilities[:, i], gold == i) for i, name in enumerate(LABELS)}}
    return result


def align_predictions(pairs, predictions):
    expected = {p["pair_id"] for p in pairs}
    by_id = {p["pair_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Predictions missing, duplicated, or outside evaluation manifest")
    if any(row["status"] != "success" or row["label"] not in (0, 1, 2) for row in predictions):
        raise ValueError("Failed predictions cannot be scored as NONE")
    return [by_id[p["pair_id"]] for p in pairs]


def evaluate(pairs, predictions):
    predictions = align_predictions(pairs, predictions)
    gold = np.array([p["label"] for p in pairs])
    probabilities = np.array([p["probabilities"] for p in predictions])
    labels = np.array([p["label"] for p in predictions])
    groups = defaultdict(list)
    for index, row in enumerate(predictions):
        feature = row["features"]
        for name in ("overall", feature["demo_cell"], "intra" if feature["sentence_distance"] == 0 else "cross",
                     f"distance_{feature['distance_bucket']}", f"label_{LABELS[gold[index]]}"):
            groups[name].append(index)
    report = {name: metrics(gold[indices], probabilities[indices], labels[indices]) for name, indices in groups.items()}
    report["test_only"] = any(p["test_only"] for p in predictions)
    report["coverage"] = 1.0
    report["label_counts"] = dict(Counter(LABELS[p["label"]] for p in pairs))
    report["complementarity"] = complementarity(pairs, predictions)
    report["input_diagnostics"] = input_diagnostics(predictions)
    return report


def input_diagnostics(predictions):
    truncated, counts, following, missing = Counter(), Counter(), Counter(), Counter()
    for row in predictions:
        for branch, expert in row["experts"].items():
            counts[branch] += 1
            truncated[branch] += expert["coverage"]["truncated"]
            plan = expert["retrieval"]
            if plan is not None:
                for label in plan["missing_labels"]:
                    missing[LABELS[label]] += 1
                if plan["demos"]:
                    following["n"] += 1
                    following["last_demo_label_matches_expert"] += int(np.argmax(expert["probabilities"])) == plan["demos"][-1]["label"]
    return {"expert_availability": {branch: count / len(predictions) for branch, count in counts.items()},
            "truncation_rate": {branch: truncated[branch] / count for branch, count in counts.items()},
            "demo_missing_labels": dict(missing), "last_demo_follow_counts": dict(following)}


def complementarity(pairs, predictions):
    coverage = Counter()
    class_coverage = defaultdict(Counter)
    weights, errors = defaultdict(list), defaultdict(list)
    for pair, row in zip(pairs, predictions):
        flags = []
        for branch, expert in row["experts"].items():
            correct = int(np.argmax(expert["probabilities"])) == pair["label"]
            flags.append(correct)
            errors[branch].append((pair["pair_id"], int(not correct)))
        name = "all_correct" if all(flags) else ("all_wrong" if not any(flags) else "mixed")
        coverage[name] += 1
        class_coverage[LABELS[pair["label"]]][name] += 1
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


def bootstrap(pairs, first, second, repeats=2000, seed=209, by_topic=False):
    first, second = align_predictions(pairs, first), align_predictions(pairs, second)
    documents = defaultdict(list)
    for i, pair in enumerate(pairs):
        key = pair["doc_id"].split("_", 1)[0] if by_topic else pair["doc_id"]
        documents[key].append(i)
    counts = []
    for indices in documents.values():
        gold = [pairs[i]["label"] for i in indices]
        row = []
        for predictions in (first, second):
            result = prf(gold, [predictions[i]["label"] for i in indices], True)
            row.append([result[k] for k in ("tp", "fp", "fn")])
        counts.append(row)
    counts = np.array(counts)
    randomizer = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        sample = counts[randomizer.integers(0, len(counts), len(counts))].sum(0)
        denominator = 2 * sample[:, 0] + sample[:, 1] + sample[:, 2]
        if np.all(denominator):
            f1 = 2 * sample[:, 0] / denominator
            differences.append(f1[1] - f1[0])
    return {"unit": "topic" if by_topic else "document", "repeats": repeats, "valid_repeats": len(differences),
            "delta_direction_f1_interval_95": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
            "seed": seed}
