"""Resumable single-model re-run of the test-set evaluation (added 2026-10-10).

Uses the CURRENT (tuned v4) SCORE_PROMPT and parser from score_articles.py
unchanged. Every successful model response is appended to a JSONL cache keyed
by (model, article id, prompt hash), so a run stopped by quota can resume
later without re-spending calls on finished articles. No fallback between
models: each model is run and reported on its own.

Usage (from the repo dir, with the venv python):
  python rerun_eval.py run gemini-flash-latest
  python rerun_eval.py run gemini-flash-lite-latest
  python rerun_eval.py run claude-sonnet-5        # optional live rerun
  python rerun_eval.py run gpt-5.1                # optional live rerun
  python rerun_eval.py seed-log                   # import per-article scores
                                                  # logged in model_comparison.txt
  python rerun_eval.py report                     # writes results_rerun_2026-10-10.txt
"""

import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from judge_client import call_anthropic, call_gemini, call_openai
from score_articles import MAX_ARTICLE_CHARS, SCORE_PROMPT, score_article

load_dotenv()

REPO_DIR = Path(__file__).parent
POOL_FILE = REPO_DIR / "training_pool.jsonl"
TEST_FILE = REPO_DIR / "labels_test.csv"
RESULTS_DIR = REPO_DIR / "results"
CACHE_FILE = RESULTS_DIR / "rerun_cache_2026-10-10.jsonl"
RESULTS_FILE = RESULTS_DIR / "results_rerun_2026-10-10.txt"
LOG_FILE = RESULTS_DIR / "model_comparison.txt"
AXES = ["scale", "relevance", "major_lab", "preference", "substance"]
PROMPT_HASH = hashlib.sha256(SCORE_PROMPT.encode("utf-8")).hexdigest()[:12]

GEMINI_PACING_SECONDS = 7  # stay under free-tier requests-per-minute
LOG_NAME_TO_MODEL = {
    "Gemini Flash": "gemini-flash-latest",
    "Gemini Flash-Lite": "gemini-flash-lite-latest",
    "GPT-5.1": "gpt-5.1",
    "Claude Sonnet 5": "claude-sonnet-5",
}


class QuotaExhausted(Exception):
    pass


def load_pool() -> dict[int, dict]:
    with open(POOL_FILE, encoding="utf-8") as f:
        return {(i := json.loads(line))["id"]: i for line in f if line.strip()}


def load_labels() -> list[dict]:
    with open(TEST_FILE, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cache() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_cache(entry: dict) -> None:
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def raw_call(model: str, prompt: str) -> str:
    if model.startswith("gemini"):
        return call_gemini(prompt, model)
    if model.startswith("gpt"):
        return call_openai(prompt)
    if model.startswith("claude"):
        return call_anthropic(prompt, model=model)
    raise ValueError(f"unknown model {model}")


def call_once_with_retries(model: str, prompt: str) -> str:
    """Retry transient 503/overload and per-minute 429s politely; raise
    QuotaExhausted on a daily-quota 429."""
    waits = [15, 30, 60, 120, 180]
    for attempt in range(len(waits) + 1):
        try:
            return raw_call(model, prompt)
        except Exception as e:
            msg = str(e)
            is_429 = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate_limit" in msg.lower()
            per_minute = "PerMinute" in msg or "per minute" in msg.lower()
            transient = "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower() or "529" in msg
            if is_429 and not per_minute:
                raise QuotaExhausted(msg[:400])
            if (transient or per_minute) and attempt < len(waits):
                print(f"    transient error ({msg[:80]!r}), retrying in {waits[attempt]}s...")
                time.sleep(waits[attempt])
                continue
            raise


def run(model: str) -> None:
    pool, labels = load_pool(), load_labels()
    done = {e["id"] for e in load_cache()
            if e["model"] == model and e["prompt_hash"] == PROMPT_HASH and e["source"] == "api"}
    todo = [int(r["id"]) for r in labels if int(r["id"]) not in done]
    print(f"{model}: prompt {PROMPT_HASH}, {len(done)}/{len(labels)} already cached, {len(todo)} to go")
    for row_id in todo:
        item = pool[row_id]
        prompt = SCORE_PROMPT.format(title=item["title"], source=item["source"], text=item["text"][:MAX_ARTICLE_CHARS])
        try:
            raw = call_once_with_retries(model, prompt)
        except QuotaExhausted as e:
            print(f"[{row_id}] QUOTA EXHAUSTED -- stopping {model}. {str(e)[:300]}")
            break
        except Exception as e:
            print(f"[{row_id}] error (skipped, not cached): {str(e)[:200]}")
            continue
        parsed = score_article(item, item["text"], caller=lambda p, r=raw: r)
        append_cache({
            "model": model, "id": row_id, "prompt_hash": PROMPT_HASH, "source": "api",
            "ts": datetime.now().isoformat(timespec="seconds"), "raw": raw,
            "scores": {a: parsed[a] for a in AXES} if parsed else None,
        })
        print(f"[{row_id}] " + (" ".join(f"{a}={parsed[a]:g}" for a in AXES) if parsed else "UNPARSEABLE (cached raw)"))
        if model.startswith("gemini"):
            time.sleep(GEMINI_PACING_SECONDS)
    final = {e["id"] for e in load_cache()
             if e["model"] == model and e["prompt_hash"] == PROMPT_HASH and e["source"] == "api"}
    print(f"{model}: {len(final)}/{len(labels)} cached for prompt {PROMPT_HASH}")


def seed_log() -> None:
    """Import per-article scores printed in model_comparison.txt (same tuned
    prompt, Aug 30 run; scores printed rounded to integers)."""
    existing = {(e["model"], e["id"], e["source"]) for e in load_cache()}
    cur, added = None, 0
    pat = re.compile(r"\[(\d+)\] " + r" ".join(rf"{a}=(\S+)\(h\S+\)" for a in AXES))
    for line in open(LOG_FILE, encoding="utf-8", errors="replace"):
        m = re.match(r"=+ (.+?) =+", line.strip())
        if m:
            cur = LOG_NAME_TO_MODEL.get(m.group(1))
            continue
        m = pat.match(line.strip())
        if m and cur and (cur, int(m.group(1)), "model_comparison.txt") not in existing:
            append_cache({
                "model": cur, "id": int(m.group(1)), "prompt_hash": PROMPT_HASH,
                "source": "model_comparison.txt", "ts": "2026-08-30", "raw": None,
                "scores": {a: float(m.group(i + 2)) for i, a in enumerate(AXES)},
            })
            added += 1
    print(f"seeded {added} entries from {LOG_FILE.name}")


def scores_for(model: str, source: str) -> dict[int, dict]:
    out = {}
    want = "model_comparison.txt" if source == "log" else source
    for e in load_cache():
        if e["model"] == model and e["prompt_hash"] == PROMPT_HASH and e["source"] == want and e["scores"]:
            out[e["id"]] = e["scores"]
    return out


def mae_table(scores: dict[int, dict], labels: dict[int, dict], ids: list[int]) -> tuple[dict, float]:
    per = {a: sum(abs(float(labels[i][a]) - scores[i][a]) for i in ids) / len(ids) for a in AXES}
    return per, sum(per.values()) / len(AXES)


def fmt(name: str, ids: list[int], scores, labels) -> str:
    if not ids:
        return f"  {name:<44} N=0"
    per, overall = mae_table(scores, labels, ids)
    return f"  {name:<44} N={len(ids):<3} " + " ".join(f"{a}={per[a]:.2f}" for a in AXES) + f"  OVERALL={overall:.2f}"


def report() -> None:
    labels = {int(r["id"]): r for r in load_labels()}
    cache = load_cache()
    lines = [
        "Re-run evaluation results -- 2026-10-10",
        f"Prompt: current tuned (v4) SCORE_PROMPT from score_articles.py, sha256[:12]={PROMPT_HASH}",
        f"Test set: {TEST_FILE.name} (n={len(labels)}). Metric: per-axis MAE vs hand labels; OVERALL = mean of 5 axis MAEs.",
        "Sources: 'api' = fresh call made 2026-10-10 (cached in rerun_cache_2026-10-10.jsonl);",
        "         'log' = per-article scores parsed from model_comparison.txt (Aug 30, same prompt, printed rounded to integers).",
        "",
    ]
    for gem in ["gemini-flash-latest", "gemini-flash-lite-latest"]:
        g = scores_for(gem, "api")
        attempted_unparseable = sum(1 for e in cache if e["model"] == gem and e["source"] == "api" and not e["scores"])
        ids = sorted(g)
        lines.append(f"== {gem} (fresh, no fallback): {len(ids)}/{len(labels)} scored, {attempted_unparseable} unparseable ==")
        lines.append(fmt(f"{gem} [api]", ids, g, labels))
        for other in ["claude-sonnet-5", "gpt-5.1", "gemini-flash-latest", "gemini-flash-lite-latest"]:
            if other == gem:
                continue
            for src in ["api", "log"]:
                o = scores_for(other, src)
                common = [i for i in ids if i in o]
                if o:
                    lines.append(fmt(f"{other} [{src}] same subset", common, o, labels))
        lines.append("")
    lines.append("== Full-set reference (all models/sources available) ==")
    for m in ["claude-sonnet-5", "gpt-5.1", "gemini-flash-lite-latest", "gemini-flash-latest"]:
        for src in ["api", "log"]:
            s = scores_for(m, src)
            if s:
                lines.append(fmt(f"{m} [{src}]", sorted(s), s, labels))
    text = "\n".join(lines) + "\n"
    RESULTS_FILE.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        run(sys.argv[2])
    elif cmd == "seed-log":
        seed_log()
    else:
        report()
