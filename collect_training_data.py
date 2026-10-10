"""Accumulates a labeled-dataset candidate pool, run once a day until it
reaches 100 readable articles. Pulls from the FULL raw candidate pool (not
just the stage-1 shortlist) so the dataset includes both good and bad
examples — needed to test whether a model/prompt correctly identifies junk,
not just ranks among already-good stories.

Two output files:
- training_pool.jsonl: full article text, for internal use by the eval script
- labels.csv: title + link only, for hand-labeling (scale/relevance/
  major_lab/preference/substance columns left blank for you to fill in)

IMPORTANT: save_labels_csv only ever APPENDS new rows — it never rewrites
existing rows, so labels you've already filled in are never touched.
"""

import csv
import json
from pathlib import Path

from article_text import fetch_article_text
from fetch_news import LOOKBACK_HOURS, TOPIC, fetch_google_news, fetch_hacker_news

TARGET_SIZE = 100
REPO_DIR = Path(__file__).parent
POOL_FILE = REPO_DIR / "training_pool.jsonl"
LABELS_FILE = REPO_DIR / "labels.csv"
LABEL_COLUMNS = ["scale", "relevance", "major_lab", "preference", "substance"]


def load_pool() -> list[dict]:
    if not POOL_FILE.exists():
        return []
    with open(POOL_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pool(pool: list[dict]) -> None:
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        for item in pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_new_labels_rows(new_items: list[dict]) -> None:
    """Only appends rows for newly-added items. Never touches existing rows,
    so labels already filled in (here or in an external spreadsheet you copy
    back in) are never at risk."""
    file_exists = LABELS_FILE.exists()
    with open(LABELS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "source", "title", "link"] + LABEL_COLUMNS)
        for item in new_items:
            writer.writerow([item["id"], item["source"], item["title"], item["link"]] + [""] * len(LABEL_COLUMNS))


def main() -> None:
    pool = load_pool()
    existing_links = {item["link"] for item in pool}
    print(f"Current pool: {len(pool)}/{TARGET_SIZE} readable articles.")

    if len(pool) >= TARGET_SIZE:
        print("Already at target size — not fetching more. Delete/trim training_pool.jsonl to collect fresh ones.")
        return

    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...")
    candidates = fetch_google_news(TOPIC, LOOKBACK_HOURS) + fetch_hacker_news(TOPIC, LOOKBACK_HOURS)
    new_candidates = [c for c in candidates if c["link"] not in existing_links]
    print(f"Got {len(candidates)} candidates, {len(new_candidates)} not already in the pool. Extracting...\n")

    new_items = []
    for c in new_candidates:
        if len(pool) + len(new_items) >= TARGET_SIZE:
            break
        text, reason = fetch_article_text(c["link"])
        if not text:
            print(f"[SKIP: {reason}] [{c['source']}] {c['title']}")
            continue

        item = {
            "id": len(pool) + len(new_items) + 1,
            "source": c["source"],
            "title": c["title"],
            "link": c["link"],
            "text": text,
        }
        new_items.append(item)
        print(f"[ADDED {item['id']}/{TARGET_SIZE}] [{c['source']}] {c['title']}")

    pool.extend(new_items)
    save_pool(pool)
    append_new_labels_rows(new_items)

    print(f"\nAdded {len(new_items)} today. Pool now: {len(pool)}/{TARGET_SIZE}.")
    if len(pool) < TARGET_SIZE:
        print("Run this again (e.g. tomorrow) to keep collecting.")
    else:
        print(f"Target reached — {LABELS_FILE} is ready for you to hand-label.")


if __name__ == "__main__":
    main()
