"""Compares judge models head-to-head on the exact same fixed (tuned) prompt,
against the held-out test set. This is the "which model is best" check, run
only after prompt tuning is finalized on the training set."""

import csv
import json
import math
from pathlib import Path

from dotenv import load_dotenv

from judge_client import SINGLE_MODEL_CALLERS
from score_articles import score_article

load_dotenv()

REPO_DIR = Path(__file__).parent
POOL_FILE = REPO_DIR / "training_pool.jsonl"
TEST_FILE = REPO_DIR / "labels_test.csv"
AXES = ["scale", "relevance", "major_lab", "preference", "substance"]


def load_pool_by_id() -> dict[int, dict]:
    with open(POOL_FILE, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return {item["id"]: item for item in items}


def load_test_labels() -> list[dict]:
    with open(TEST_FILE, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if std_x == 0 or std_y == 0:
        return float("nan")
    return cov / (std_x * std_y)


def evaluate_model(model_name: str, caller, pool: dict, test_labels: list[dict]) -> dict:
    print(f"\n========== {model_name} ==========")
    human = {axis: [] for axis in AXES}
    model = {axis: [] for axis in AXES}
    errors = 0

    for row in test_labels:
        row_id = int(row["id"])
        item = pool.get(row_id)
        if not item:
            errors += 1
            continue
        try:
            result = score_article(item, item["text"], caller=caller)
        except Exception as e:
            print(f"[{row_id}] error: {e}")
            errors += 1
            continue
        if not result:
            print(f"[{row_id}] unparseable response")
            errors += 1
            continue
        for axis in AXES:
            human[axis].append(float(row[axis]))
            model[axis].append(result[axis])
        print(f"[{row_id}] " + " ".join(f"{a}={result[a]:.0f}(h{row[a]})" for a in AXES))

    n = len(test_labels) - errors
    per_axis = {}
    for axis in AXES:
        mae = sum(abs(h - m) for h, m in zip(human[axis], model[axis])) / n if n else float("nan")
        corr = pearson(human[axis], model[axis]) if n else float("nan")
        per_axis[axis] = (mae, corr)
    overall_mae = sum(v[0] for v in per_axis.values()) / len(AXES)
    return {"n": n, "errors": errors, "per_axis": per_axis, "overall_mae": overall_mae}


def main() -> None:
    pool = load_pool_by_id()
    test_labels = load_test_labels()
    print(f"Comparing {len(SINGLE_MODEL_CALLERS)} models against {len(test_labels)} test articles...")

    results = {}
    for model_name, caller in SINGLE_MODEL_CALLERS.items():
        results[model_name] = evaluate_model(model_name, caller, pool, test_labels)

    print("\n\n############ Model comparison (test set) ############\n")
    header = f"{'Model':<20} {'N':<5} " + " ".join(f"{a:<16}" for a in AXES) + f"{'Overall MAE':<12}"
    print(header)
    for model_name, r in results.items():
        row = f"{model_name:<20} {r['n']:<5} "
        for axis in AXES:
            mae, corr = r["per_axis"][axis]
            row += f"{mae:.2f}/{corr:.2f}      "
        row += f"{r['overall_mae']:.2f}"
        print(row)
    print("\n(format per axis: MAE/correlation)")


if __name__ == "__main__":
    main()
