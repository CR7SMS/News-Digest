"""Milestone 2: dedupe + score the fetched headlines with Gemini as judge,
one call for the whole batch, and print a shortlist. This is stage 1 of a
two-stage pipeline: cheap headline-only filtering down to a shortlist here,
then (stage 2, separate milestone) full-article per-dimension scoring on just
the shortlist survivors."""

import re

from dotenv import load_dotenv

from fetch_news import LOOKBACK_HOURS, TOPIC, fetch_google_news, fetch_hacker_news
from judge_client import call_with_fallback

load_dotenv()

TOP_N = 20  # stage-1 shortlist size, not the final digest count

JUDGE_PROMPT = """You are curating a personal AI news digest for a Data Scientist who works on AI/LLM evaluation (LLM-as-judge, benchmarking, model reliability).

Below is a numbered list of headlines from the last {lookback_hours} hours. Many are duplicates or near-duplicates of the same underlying story from different outlets, and many are low-substance filler (SEO listicles, stock-picking punditry, legal-drama trivia) rather than real developments.

Your job:
1. Group near-duplicate stories about the same event together — treat them as one story. Pick the clearest headline as the representative ID, and list every OTHER headline number covering the same story as its alt IDs (different outlets often cover the same event — these alt links matter because the representative one might turn out to be paywalled later).
2. Score each surviving distinct story on these dimensions (higher = better fit for this reader):
   - Scale/significance of the development
   - Relevance to LLM-as-judge / AI evaluation specifically
   - Involves a major lab (OpenAI, Anthropic, Google, Amazon, Meta)
   - Personal preference — how much this specific reader would want to read it, independent of the other scores
   - Substance (a real development, not SEO/opinion/stock-pick filler — score this LOW for listicles, stock predictions, legal trivia, celebrity-AI stories)
3. Return the top {top_n} stories overall, ranked best first — this is a first-pass shortlist (headline-only), not the final ranking.

Headlines:
{headline_list}

Respond with EXACTLY one line per ranked story, in this format and nothing else:
RANK=<n> | ID=<representative headline number> | ALT_IDS=<comma-separated other headline numbers covering the same story, or "none"> | SCORE=<0-10 overall> | REASON=<one short sentence>
"""


def build_headline_list(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, start=1):
        lines.append(f"{i}. [{a['source']}] {a['title']}")
    return "\n".join(lines)


def judge_and_rank(articles: list[dict], caller=call_with_fallback) -> str:
    prompt = JUDGE_PROMPT.format(
        lookback_hours=LOOKBACK_HOURS,
        top_n=TOP_N,
        headline_list=build_headline_list(articles),
    )
    return caller(prompt)


def parse_ranking(response_text: str) -> list[dict]:
    pattern = (
        r"RANK=(\d+)\s*\|\s*ID=(\d+)\s*\|\s*ALT_IDS=([^|]*)\s*\|\s*SCORE=([\d.]+)\s*\|\s*REASON=(.+)"
    )
    picks = []
    for line in response_text.strip().splitlines():
        m = re.match(pattern, line.strip())
        if m:
            alt_ids_raw = m.group(3).strip()
            if alt_ids_raw.lower() == "none" or not alt_ids_raw:
                alt_ids = []
            else:
                alt_ids = [int(x.strip()) for x in alt_ids_raw.split(",") if x.strip().isdigit()]
            picks.append(
                {
                    "rank": int(m.group(1)),
                    "id": int(m.group(2)),
                    "alt_ids": alt_ids,
                    "score": float(m.group(4)),
                    "reason": m.group(5).strip(),
                }
            )
    return picks


def main() -> None:
    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...")
    articles = fetch_google_news(TOPIC, LOOKBACK_HOURS) + fetch_hacker_news(TOPIC, LOOKBACK_HOURS)
    print(f"Got {len(articles)} candidate headlines. Asking Gemini to dedupe + rank...\n")

    raw_response = judge_and_rank(articles)
    picks = parse_ranking(raw_response)

    if not picks:
        print("Couldn't parse a ranking out of the response. Raw output:\n")
        print(raw_response)
        return

    print(f"############ Shortlist ({len(picks)}) ############\n")
    for p in picks:
        article = articles[p["id"] - 1]
        alt_note = f"  [+{len(p['alt_ids'])} alt link(s)]" if p["alt_ids"] else ""
        print(f"{p['rank']}. [{article['source']}] {article['title']}  (score {p['score']}){alt_note}")
        print(f"   {p['reason']}")
        print(f"   {article['link']}")
        print()


if __name__ == "__main__":
    main()
