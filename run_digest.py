"""Single entrypoint for the daily digest: fetch -> stage-1 headline shortlist
-> full-text extraction -> stage-2 per-article scoring -> dedupe -> top N ->
render (and optionally email) the digest.

Usage:
  python run_digest.py            # run everything and send the email (if configured)
  python run_digest.py --no-send  # run everything, save output/ files, don't send

Config comes from environment variables. Locally they're loaded from .env
(python-dotenv never overrides variables that are already set, so in GitHub
Actions the repository secrets win).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from article_text import fetch_article
from email_digest import render_html, render_text, send_email, subject_for
from fetch_news import LOOKBACK_HOURS, TOPIC, fetch_google_news, fetch_hacker_news
from judge_client import call_production
from rank_news import judge_and_rank, parse_ranking
from score_articles import FINAL_N, score_article

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
TITLE_SIMILARITY = 0.85  # normalized-title ratio at/above which two stories count as the same


# ---------------------------------------------------------------- dedupe

def normalize_url(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}".lower()


def display_title(title: str, outlet: str) -> str:
    """Google News titles end in ' - Outlet'; drop that for display."""
    suffix = f" - {outlet}"
    return title[: -len(suffix)] if outlet and title.endswith(suffix) else title


def normalize_title(title: str) -> str:
    t = title.lower()
    if " - " in t:  # drop a trailing " - Outlet" if present
        head, tail = t.rsplit(" - ", 1)
        if len(tail.split()) <= 4:
            t = head
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_duplicate(story: dict, kept: list[dict]) -> dict | None:
    for k in kept:
        if story["url_key"] == k["url_key"]:
            return k
        if SequenceMatcher(None, story["title_key"], k["title_key"]).ratio() >= TITLE_SIMILARITY:
            return k
    return None


def dedupe(stories: list[dict]) -> list[dict]:
    """Keep the first occurrence of each story (callers sort best-first)."""
    kept = []
    for s in stories:
        dup = is_duplicate(s, kept)
        if dup:
            print(f"  [dedupe] dropped duplicate: {s['title']}  (same as: {dup['title']})")
            continue
        kept.append(s)
    return kept


# ---------------------------------------------------------------- pipeline

def build_digest(final_n: int = FINAL_N, caller=call_production) -> list[dict]:
    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...")
    articles = fetch_google_news(TOPIC, LOOKBACK_HOURS) + fetch_hacker_news(TOPIC, LOOKBACK_HOURS)
    print(f"Got {len(articles)} candidate headlines. Stage 1: dedupe + headline shortlist...")

    picks = parse_ranking(judge_and_rank(articles, caller=caller))
    picks = [p for p in picks if 0 < p["id"] <= len(articles)]
    print(f"Shortlist: {len(picks)} stories. Stage 2: extracting full text and scoring...\n")

    scored: list[dict] = []
    for p in picks:
        primary = articles[p["id"] - 1]
        candidates = [primary] + [articles[i - 1] for i in p["alt_ids"] if 0 < i <= len(articles)]

        text = used = resolved = None
        attempts = []
        for c in candidates:
            text, reason, resolved = fetch_article(c["link"])
            attempts.append(f"[{c.get('outlet', c['source'])}] {reason}")
            if text:
                used = c
                break

        if not text:
            print(f"[SKIP unreadable] {primary['title']}")
            for a in attempts:
                print(f"    tried {a}")
            continue

        outlet = used.get("outlet") or used["source"]
        story = {
            "title": display_title(used["title"], outlet),
            "outlet": outlet,
            "url": resolved or used["link"],
            "aggregator": used["source"],
        }
        story["url_key"] = normalize_url(story["url"])
        story["title_key"] = normalize_title(story["title"])

        # Dedupe before scoring too: two stage-1 clusters can resolve to the
        # same article (e.g. via an alt link), no point paying to score it twice.
        dup = is_duplicate(story, scored)
        if dup:
            print(f"[SKIP duplicate] {story['title']}  (same as: {dup['title']})")
            continue

        result = score_article(used, text, caller=caller)
        if not result:
            print(f"[SKIP unparseable score] {story['title']}")
            continue

        story.update(result)
        scored.append(story)
        print(f"[SCORED {result['overall']:.1f}] [{outlet}] {story['title']}")

    scored.sort(key=lambda s: s["overall"], reverse=True)
    top = dedupe(scored)[:final_n]
    for s in top:
        s.pop("url_key", None)
        s.pop("title_key", None)
    return top


def print_digest(stories: list[dict]) -> None:
    print(f"\n############ Final digest (top {len(stories)}) ############\n")
    print(render_text(stories, datetime.now()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and email the daily AI news digest.")
    parser.add_argument("--no-send", action="store_true", help="build and save the digest but don't email it")
    args = parser.parse_args()

    try:  # don't crash on non-ASCII titles when stdout is a Windows code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stories = build_digest()
    if not stories:
        print("No readable, scorable stories today; nothing to send.")
        return 1

    now = datetime.now()
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = f"digest_{now:%Y-%m-%d}"
    html_path = OUTPUT_DIR / f"{stem}.html"
    html_path.write_text(render_html(stories, now), encoding="utf-8")
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(stories, indent=2, ensure_ascii=False), encoding="utf-8")

    print_digest(stories)
    print(f"Saved {html_path}")

    if args.no_send:
        print("--no-send given: not emailing.")
    else:
        send_email(subject_for(now), render_html(stories, now), render_text(stories, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
