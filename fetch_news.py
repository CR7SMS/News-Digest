"""Milestone 1 (+ second source): pull AI news links from Google News RSS and
Hacker News, print them. No judging/scoring yet — just prove we can fetch a
real candidate list."""

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import feedparser
import requests

TOPIC = "AI"
LOOKBACK_HOURS = 24


def fetch_google_news(topic: str, lookback_hours: int) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    results = []
    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published >= cutoff:
            results.append(
                {
                    "source": "Google News",
                    "title": entry.title,
                    "link": entry.link,
                    "published": published,
                    # the publisher (e.g. "Reuters"), for display only; "source"
                    # stays the aggregator name since the scoring prompt uses it
                    "outlet": (entry.get("source") or {}).get("title") or "Google News",
                }
            )
    return results


def _domain(link: str) -> str:
    host = urlparse(link).netloc.lower().removeprefix("www.")
    return "Hacker News" if host in ("", "news.ycombinator.com") else host


def fetch_hacker_news(topic: str, lookback_hours: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cutoff_ts = int(cutoff.timestamp())

    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "query": topic,
        "tags": "story",
        "numericFilters": f"created_at_i>{cutoff_ts}",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = []
    for hit in data.get("hits", []):
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
        results.append(
            {
                "source": "Hacker News",
                "title": hit["title"],
                "link": link,
                "published": datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc),
                "points": hit.get("points", 0),
                "outlet": _domain(link),
            }
        )
    return results


def main() -> None:
    print(f"Fetching '{TOPIC}' news from the last {LOOKBACK_HOURS}h...\n")

    google_results = fetch_google_news(TOPIC, LOOKBACK_HOURS)
    hn_results = fetch_hacker_news(TOPIC, LOOKBACK_HOURS)

    all_results = google_results + hn_results
    all_results.sort(key=lambda r: r["published"], reverse=True)

    print(f"Google News: {len(google_results)} articles")
    print(f"Hacker News: {len(hn_results)} stories")
    print(f"Total: {len(all_results)}\n")

    for r in all_results:
        extra = f" ({r['points']} pts)" if "points" in r else ""
        print(f"[{r['source']}{extra}] {r['title']}")
        print(f"  {r['link']}")
        print(f"  {r['published'].strftime('%Y-%m-%d %H:%M UTC')}")
        print()


if __name__ == "__main__":
    main()
