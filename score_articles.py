"""Milestone 3 (stage 2): "hardcore" scoring — one judge call per article,
on the full extracted text, returning 5 separate per-dimension scores instead
of stage 1's single blended headline-only score. Only runs on articles that
survived stage 2 extraction (readable, not paywalled/blocked)."""

from dotenv import load_dotenv

from article_text import fetch_article_text
from fetch_news import LOOKBACK_HOURS, TOPIC, fetch_google_news, fetch_hacker_news
from judge_client import call_with_fallback
from rank_news import judge_and_rank, parse_ranking

load_dotenv()

FINAL_N = 10
MAX_ARTICLE_CHARS = 4000  # plenty for scoring; keeps prompt/cost small

SCORE_PROMPT = """You are scoring a single news article for a personal AI news digest. The reader is a Data Scientist who works on AI/LLM evaluation (LLM-as-judge, benchmarking, model reliability).

Title: {title}
Source: {source}

Article text:
{text}

Score this article 0-10 on each dimension:
- SCALE: how significant/large-scale is this development in the AI industry? If the article isn't genuinely about an AI/LLM development, or is only a minor/routine mention, scale is near 0 — don't default to a generous middle score just because something AI-adjacent is mentioned.
- RELEVANCE: how relevant is this to AI/LLM evaluation broadly — LLM-as-judge, benchmarking, model reliability, safety evaluation, or trustworthiness of AI systems? Don't require the words "judge" or "evaluation" to appear — a story about a model's real-world reliability, a new benchmark, or a trust/safety incident all count as relevant even if not phrased as "evaluation."
- MAJOR_LAB: two tiers. Top tier — OpenAI, Anthropic, Google/DeepMind, Meta, Amazon — can score up to 10 if central to the story (recognize them by product names too: ChatGPT/GPT→OpenAI, Claude→Anthropic, Gemini→Google, Llama→Meta). Second tier — other real AI-industry players (Nvidia, Microsoft, Oracle, Salesforce, Alibaba, and similar large AI-adjacent/infrastructure/enterprise companies) — cap around 5-7 even when they're the central subject, since they're real but not the frontier labs this reader's work centers on. 0 if no such company is involved at all.
- PREFERENCE: personal interest — how much would this specific reader want to read this. Score high for: new AI developments/model releases, interesting or novel use cases, stories about AI performance or capability gaps (things not working as claimed), AI-related hacking/security incidents, and evaluation-related developments. Score low for generic policy/political AI coverage, stock/business opinion pieces, and human-interest or celebrity-adjacent AI mentions.
- SUBSTANCE: how much real, concrete, on-topic AI information is actually here? Being well-written or factual is NOT enough — if the article isn't genuinely about an AI/LLM development (even if it's a real, well-reported story about something else), substance is near 0 regardless of quality. Among articles that ARE about AI, most day-to-day coverage is still low substance (0-2): routine corporate PR, stock commentary, opinion/reaction pieces, generic "AI in X" trend pieces, and quote round-ups. Reserve 7+ for something that teaches a genuinely new, concrete, actionable fact about an AI system, model, or method. Default low, not high.

Respond with EXACTLY this format and nothing else:
SCALE=<0-10> | RELEVANCE=<0-10> | MAJOR_LAB=<0-10> | PREFERENCE=<0-10> | SUBSTANCE=<0-10> | REASON=<one short sentence>
"""


def score_article(article: dict, text: str, caller=call_with_fallback) -> dict | None:
    prompt = SCORE_PROMPT.format(
        title=article["title"],
        source=article["source"],
        text=text[:MAX_ARTICLE_CHARS],
    )
    response = caller(prompt)

    import re

    pattern = (
        r"SCALE=([\d.]+)\s*\|\s*RELEVANCE=([\d.]+)\s*\|\s*MAJOR_LAB=([\d.]+)\s*\|\s*"
        r"PREFERENCE=([\d.]+)\s*\|\s*SUBSTANCE=([\d.]+)\s*\|\s*REASON=(.+)"
    )
    m = re.search(pattern, response.strip())
    if not m:
        return None

    scale, relevance, major_lab, preference, substance = (float(m.group(i)) for i in range(1, 6))
    overall = (scale + relevance + major_lab + preference + substance) / 5

    return {
        "scale": scale,
        "relevance": relevance,
        "major_lab": major_lab,
        "preference": preference,
        "substance": substance,
        "overall": overall,
        "reason": m.group(6).strip(),
    }


def main() -> None:
    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...")
    articles = fetch_google_news(TOPIC, LOOKBACK_HOURS) + fetch_hacker_news(TOPIC, LOOKBACK_HOURS)
    print(f"Got {len(articles)} candidates. Getting stage-1 shortlist...\n")

    raw_response = judge_and_rank(articles)
    picks = parse_ranking(raw_response)

    print(f"Shortlist: {len(picks)} stories. Extracting full text (trying alt links if the primary is blocked)...\n")

    scored = []
    for p in picks:
        primary = articles[p["id"] - 1]
        candidates = [primary] + [articles[i - 1] for i in p["alt_ids"] if 0 < i <= len(articles)]

        text = None
        used_article = None
        attempts = []
        for candidate in candidates:
            text, reason = fetch_article_text(candidate["link"])
            attempts.append(f"[{candidate['source']}] {reason}")
            if text:
                used_article = candidate
                break

        if not text:
            print(f"[SKIP all links unreadable] [{primary['source']}] {primary['title']}")
            for a in attempts:
                print(f"   tried {a}")
            continue

        if used_article is not primary:
            print(f"  (primary [{primary['source']}] was blocked — used alt [{used_article['source']}] instead)")

        result = score_article(used_article, text)
        if not result:
            print(f"[SKIP unparseable score] [{used_article['source']}] {used_article['title']}")
            continue

        scored.append((used_article, result))
        print(f"[SCORED {result['overall']:.1f}] [{used_article['source']}] {used_article['title']}")

    scored.sort(key=lambda x: x[1]["overall"], reverse=True)
    top = scored[:FINAL_N]

    print(f"\n############ Final digest (top {len(top)}) ############\n")
    for i, (a, r) in enumerate(top, start=1):
        print(f"{i}. [{a['source']}] {a['title']}  (overall {r['overall']:.1f})")
        print(
            f"   scale={r['scale']:.0f} relevance={r['relevance']:.0f} "
            f"major_lab={r['major_lab']:.0f} preference={r['preference']:.0f} substance={r['substance']:.0f}"
        )
        print(f"   {r['reason']}")
        print(f"   {a['link']}")
        print()


if __name__ == "__main__":
    main()
