"""Full-article text extraction. A failed/too-short extraction (paywall,
bot-blocked, teaser-only) is treated as a filtering signal, not an error to
route around — if we can't read it for free, it's not useful for the digest."""

import trafilatura
from googlenewsdecoder import gnewsdecoder

MIN_LENGTH = 200  # shorter than this is almost always a paywall teaser stub


def resolve_url(url: str) -> tuple[str, str | None]:
    """Google News RSS links are redirect/interstitial pages, not the real
    article — decode them to the actual publisher URL first. Returns
    (url_to_fetch, error_or_None)."""
    if "news.google.com" not in url:
        return url, None

    result = gnewsdecoder(url, interval=1)
    if result.get("status") and result.get("decoded_url"):
        return result["decoded_url"], None
    return url, f"could not decode Google News redirect: {result.get('message', 'unknown error')}"


def fetch_article_text(url: str) -> tuple[str | None, str]:
    """Returns (text_or_None, reason). reason is always populated, even on success."""
    text, reason, _ = fetch_article(url)
    return text, reason


def fetch_article(url: str) -> tuple[str | None, str, str]:
    """Like fetch_article_text, but also returns the resolved publisher URL
    (useful for linking and for de-duplicating stories)."""
    real_url, decode_error = resolve_url(url)
    text, reason = _extract(real_url, decode_error)
    return text, reason, real_url


def _extract(real_url: str, decode_error: str | None) -> tuple[str | None, str]:
    if decode_error:
        return None, decode_error

    try:
        downloaded = trafilatura.fetch_url(real_url)
    except Exception as e:
        return None, f"fetch exception: {e}"

    if not downloaded:
        return None, "fetch_url returned nothing (network error, timeout, or blocked)"

    if "news.google.com" in real_url and "<base href=\"https://news.google.com/\">" in downloaded:
        return None, "got Google's own redirect/interstitial page, not the publisher article"

    try:
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    except Exception as e:
        return None, f"extract exception: {e}"

    if not text:
        return None, "extract() found no article content (likely paywalled/JS-rendered/bot-blocked)"

    if len(text) < MIN_LENGTH:
        return None, f"extracted only {len(text)} chars (likely a paywall teaser stub)"

    return text, "ok"
