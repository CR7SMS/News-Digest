"""Shared judge-call logic: try Gemini flash (free), retry on transient
overload, then try Gemini flash-lite (separate daily quota — lite models
tend to get a higher free allowance), then finally fall back to GPT. Used by
both stage 1 (headline shortlist) and stage 2 (per-article scoring)."""

import time

import anthropic
from google import genai
from openai import OpenAI

GOOGLE_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # each has
# its own separate daily quota, so if the first is exhausted the second is a
# real second chance, not just a retry of the same limit
OPENAI_MODEL = "gpt-5.1"  # last-resort fallback — covered by OpenAI's free
# 250K tokens/day allowance (data-sharing opt-in), so it's effectively free
# at this call volume
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 10


def call_gemini(prompt: str, model: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def call_openai(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_with_fallback(prompt: str, google_models: list[str] | None = None) -> str:
    last_error = None

    for model in google_models or GOOGLE_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                return call_gemini(prompt, model)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    print(f"  {model} daily quota exhausted, trying next option...")
                    break  # don't retry a quota error, move to the next model
                if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                        print(f"  {model} overloaded, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                        time.sleep(wait)
                        continue
                break

    print(f"  All Gemini options unavailable ({last_error}). Falling back to GPT ({OPENAI_MODEL})...")
    return call_openai(prompt)


# Production chain for the daily digest (run_digest.py): Flash-Lite first,
# because it's on the free tier and its test-set MAE with the tuned prompt
# (1.61) is close to the best model tested (Claude Sonnet 5, 1.55). The
# default GOOGLE_MODELS order above is left as-is so evaluation scripts that
# call call_with_fallback() keep behaving exactly as before.
PRODUCTION_GOOGLE_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]


def call_production(prompt: str) -> str:
    """Flash-Lite -> Flash -> GPT-5.1. Used only by the daily digest."""
    return call_with_fallback(prompt, google_models=PRODUCTION_GOOGLE_MODELS)


def call_anthropic(prompt: str, model: str = "claude-sonnet-5") -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("No text block found in response")


def with_retry(caller, prompt: str):
    """Retry a single fixed model on transient overload, without falling
    through to a different model — for controlled single-model comparisons."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return caller(prompt)
        except Exception as e:
            last_error = e
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    print(f"    overloaded, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
            raise last_error
    raise last_error


# Named single-model callers, no fallback — for comparing models head-to-head
# on the exact same fixed prompt.
SINGLE_MODEL_CALLERS = {
    "Gemini Flash": lambda p: with_retry(lambda pr: call_gemini(pr, "gemini-flash-latest"), p),
    "Gemini Flash-Lite": lambda p: with_retry(lambda pr: call_gemini(pr, "gemini-flash-lite-latest"), p),
    "GPT-5.1": lambda p: with_retry(call_openai, p),
    "Claude Sonnet 5": lambda p: with_retry(call_anthropic, p),
}
