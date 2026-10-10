"""Evaluates the current judge prompt/model against hand-labeled test data.
Runs stage-2 per-dimension scoring on each test article's full text, compares
to the hand-labels per axis, and reports mean absolute error (MAE) and
Pearson correlation per dimension. This is the baseline number future
prompt/model changes get compared against.
"""

import csv
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

from score_articles import score_article

load_dotenv()

REPO_DIR = Path(__file__).parent
POOL_FILE = REPO_DIR / "training_pool.jsonl"
AXES = ["scale", "relevance", "major_lab", "preference", "substance"]


def load_pool_by_id() -> dict[int, dict]:
    with open(POOL_FILE, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return {item["id"]: item for item in items}


def load_test_labels(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
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


def main() -> None:
    test_file = REPO_DIR / (sys.argv[1] if len(sys.argv) > 1 else "labels_test.csv")
    pool = load_pool_by_id()
    test_labels = load_test_labels(test_file)

    print(f"Evaluating current prompt/model against {len(test_labels)} test articles ({test_file.name})...\n")

    human = {axis: [] for axis in AXES}
    model = {axis: [] for axis in AXES}
    errors = 0

    for row in test_labels:
        row_id = int(row["id"])
        item = pool.get(row_id)
        if not item:
            print(f"[SKIP {row_id}: not found in training_pool.jsonl] {row['title']}")
            errors += 1
            continue

        result = score_article(item, item["text"])
        if not result:
            print(f"[SKIP {row_id}: unparseable model response] {row['title']}")
            errors += 1
            continue

        for axis in AXES:
            human[axis].append(float(row[axis]))
            model[axis].append(result[axis])

        diffs = " ".join(f"{axis}={result[axis]:.0f}(human {row[axis]})" for axis in AXES)
        print(f"[{row_id}] {diffs}")

    n = len(test_labels) - errors
    print(f"\n############ Baseline results (n={n}) ############\n")
    print(f"{'Axis':<12} {'MAE':<8} {'Correlation':<12}")
    total_mae = 0.0
    for axis in AXES:
        mae = sum(abs(h - m) for h, m in zip(human[axis], model[axis])) / n
        corr = pearson(human[axis], model[axis])
        total_mae += mae
        print(f"{axis:<12} {mae:<8.2f} {corr:<12.2f}")
    print(f"\nOverall mean MAE across all axes: {total_mae / len(AXES):.2f}")


if __name__ == "__main__":
    main()
