"""Checkpoint before building stage-2 scoring: run the stage-1 shortlist,
then attempt full-article extraction on each. See what's actually readable
before designing the per-dimension scoring prompts on top of it."""

from article_text import fetch_article_text
from fetch_news import LOOKBACK_HOURS, TOPIC, fetch_google_news, fetch_hacker_news
from rank_news import judge_and_rank, parse_ranking


def main() -> None:
    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...")
    articles = fetch_google_news(TOPIC, LOOKBACK_HOURS) + fetch_hacker_news(TOPIC, LOOKBACK_HOURS)
    print(f"Got {len(articles)} candidates. Getting stage-1 shortlist...\n")

    raw_response = judge_and_rank(articles)
    picks = parse_ranking(raw_response)
    shortlist = [articles[p["id"] - 1] for p in picks]

    print(f"Shortlist: {len(shortlist)} articles. Attempting full-text extraction...\n")

    readable = []
    unreadable = []
    for a in shortlist:
        text, reason = fetch_article_text(a["link"])
        if text:
            readable.append((a, text))
            print(f"[READABLE  {len(text):>5} chars] [{a['source']}] {a['title']}")
        else:
            unreadable.append((a, reason))
            print(f"[FAILED] [{a['source']}] {a['title']}")
            print(f"         reason: {reason}")
            print(f"         link:   {a['link']}")

    print(f"\n############ Summary ############")
    print(f"Readable:   {len(readable)}/{len(shortlist)}")
    print(f"Unreadable: {len(unreadable)}/{len(shortlist)} (filtered out — paywalled/blocked/too short)")


if __name__ == "__main__":
    main()
