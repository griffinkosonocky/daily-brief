#!/usr/bin/env python3
"""Update the static news ticker without using an LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
TICKER_FILE = SITE_DIR / "ticker.json"

FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("AP", "https://apnews.com/hub/ap-top-news?output=rss"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    (
        "Reuters",
        "https://news.google.com/rss/search?q=Reuters%20world%20news&hl=en-US&gl=US&ceid=US:en",
    ),
]

FALLBACK_HEADLINES = [
    {"source": "BBC", "text": "World headlines will appear here after the first successful RSS update."},
    {"source": "AP", "text": "The ticker refreshes hourly from BBC, AP, NPR, and Reuters via Google News RSS."},
    {"source": "NPR", "text": "If a feed is temporarily unavailable, the last good ticker is kept."},
]


@dataclass
class Headline:
    source: str
    text: str
    published_at: str
    sort_time: datetime


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    # Google News search feeds often append " - Reuters"; keep the ticker format
    # consistent by using the explicit source label instead.
    return re.sub(r"\s+-\s+Reuters\s*$", "", title).strip()


def normalized(title: str) -> str:
    words = re.findall(r"[a-z0-9]+", title.lower())
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as"}
    return " ".join(word for word in words if word not in stop)


def too_similar(candidate: str, existing: Iterable[str]) -> bool:
    candidate_norm = normalized(candidate)
    for title in existing:
        existing_norm = normalized(title)
        if not candidate_norm or not existing_norm:
            continue
        if candidate_norm == existing_norm:
            return True
        if SequenceMatcher(None, candidate_norm, existing_norm).ratio() >= 0.86:
            return True
    return False


def parse_datetime(entry: dict) -> datetime:
    for key in ("published", "updated"):
        value = entry.get(key)
        if value:
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except (TypeError, ValueError):
                pass
    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct:
        return datetime(*parsed_struct[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def fetch_headlines(limit: int = 15) -> list[dict]:
    try:
        import feedparser  # type: ignore
    except ModuleNotFoundError:
        print("feedparser is not installed; using fallback ticker data.")
        return load_existing_or_fallback()

    candidates: list[Headline] = []
    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as exc:  # pragma: no cover - defensive around RSS IO
            print(f"Ticker feed failed for {source}: {exc}")
            continue

        if getattr(feed, "bozo", False):
            print(f"Ticker feed warning for {source}: {getattr(feed, 'bozo_exception', 'unknown')}")

        for entry in feed.entries[:30]:
            title = clean_title(entry.get("title", ""))
            if not title:
                continue
            published = parse_datetime(entry)
            candidates.append(
                Headline(
                    source=source,
                    text=title,
                    published_at=published.replace(microsecond=0).isoformat(),
                    sort_time=published,
                )
            )

    candidates.sort(key=lambda item: item.sort_time, reverse=True)
    headlines: list[dict] = []
    seen_titles: list[str] = []
    for item in candidates:
        if too_similar(item.text, seen_titles):
            continue
        headlines.append(
            {
                "source": item.source,
                "text": item.text,
                "published_at": item.published_at,
            }
        )
        seen_titles.append(item.text)
        if len(headlines) >= limit:
            break

    if not headlines:
        return load_existing_or_fallback()
    return headlines


def load_existing_or_fallback() -> list[dict]:
    if TICKER_FILE.exists():
        try:
            existing = json.loads(TICKER_FILE.read_text(encoding="utf-8"))
            headlines = existing.get("headlines", [])
            if headlines:
                return headlines
        except json.JSONDecodeError:
            pass
    return FALLBACK_HEADLINES


def write_if_changed(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == new_text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def update_ticker() -> bool:
    headlines = fetch_headlines()
    payload = {
        "updated_at": now_utc(),
        "headlines": headlines,
    }
    # If the headline list is unchanged, preserve the old updated_at so the
    # hourly workflow can skip a pointless commit.
    if TICKER_FILE.exists():
        try:
            existing = json.loads(TICKER_FILE.read_text(encoding="utf-8"))
            if existing.get("headlines") == headlines:
                print("Ticker headlines unchanged.")
                return False
        except json.JSONDecodeError:
            pass
    changed = write_if_changed(TICKER_FILE, payload)
    print("Ticker updated." if changed else "Ticker unchanged.")
    return changed


if __name__ == "__main__":
    update_ticker()
