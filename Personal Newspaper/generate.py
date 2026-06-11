#!/usr/bin/env python3
"""Build The Daily Brief as static files for GitHub Pages."""

from __future__ import annotations

import copy
import html
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ticker import update_ticker

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
ARCHIVE_DIR = SITE_DIR / "archive"
TOPICS_FILE = ROOT / "topics.json"
TRIVIA_FILE = ROOT / "trivia_history.json"
STATE_FILE = ROOT / "state.json"

CENTRAL = ZoneInfo("America/Chicago")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

ARTICLE_MODEL = os.getenv("ANTHROPIC_ARTICLE_MODEL", "claude-sonnet-4-6")
MARKETS_MODEL = os.getenv("ANTHROPIC_MARKETS_MODEL", "claude-sonnet-4-6")
TRIVIA_MODEL = os.getenv("ANTHROPIC_TRIVIA_MODEL", "claude-haiku-4-5")
TOPICS_MODEL = os.getenv("ANTHROPIC_TOPICS_MODEL", "claude-haiku-4-5")

SPORT_ROTATION = ["NFL", "Soccer", "Golf", "NBA", "MLB", "Wildcard"]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Could not parse {path.name}: {exc}. Using default.")
        return copy.deepcopy(default)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def today_central() -> datetime:
    return datetime.now(CENTRAL)


def iso_now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def human_date(day: datetime) -> str:
    return day.strftime("%A, %B ") + str(day.day) + day.strftime(", %Y")


def compact_ct_timestamp(now: datetime | None = None) -> str:
    now = now or today_central()
    hour = now.strftime("%I").lstrip("0") or "0"
    return now.strftime("%b ") + str(now.day) + now.strftime(f", %Y at {hour}:%M %p CT")


def anthropic_message(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    system: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    timeout: int = 180,
) -> tuple[str, list[dict], dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic API request failed: {exc}") from exc

    text_parts: list[str] = []
    citations: list[dict] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
            for citation in block.get("citations", []) or []:
                url = citation.get("url")
                title = citation.get("title")
                if url and title and not any(item.get("url") == url for item in citations):
                    citations.append({"title": title, "url": url})

    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise RuntimeError("Anthropic API returned no text content.")
    return text, citations, data.get("usage", {})


def extract_json_block(text: str) -> tuple[dict | list | None, str]:
    pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    json_like_matches = []
    for match in reversed(matches):
        raw = match.group(1).strip()
        if raw.startswith(("{", "[")):
            json_like_matches.append(match)

    for match in json_like_matches:
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            cleaned = (text[: match.start()] + text[match.end() :]).strip()
            return data, cleaned
        except json.JSONDecodeError:
            continue
    if json_like_matches:
        match = json_like_matches[0]
        cleaned = (text[: match.start()] + text[match.end() :]).strip()
        return None, cleaned
    return None, text.strip()


def parse_json_object(text: str) -> dict | None:
    data, _ = extract_json_block(text)
    if isinstance(data, dict):
        return data
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    return None


def inline_markdown(value: str) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_html(text: str) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("#"):
            flush_paragraph()
            heading = line.lstrip("#").strip()
            level = 2 if raw_line.startswith("##") else 3
            blocks.append(f"<h{level}>{inline_markdown(heading)}</h{level}>")
            continue
        if line.lower().startswith("further reading:"):
            flush_paragraph()
            label, _, rest = line.partition(":")
            blocks.append(
                f'<p class="further-reading"><strong>{html.escape(label)}:</strong> '
                f"{inline_markdown(rest.strip())}</p>"
            )
            continue
        paragraph.append(line)
    flush_paragraph()
    return "\n".join(blocks)


def validate_locations(raw_locations: Any) -> list[dict]:
    if not isinstance(raw_locations, list):
        return []
    locations: list[dict] = []
    allowed_roles = {"capital", "battle", "city", "site"}
    for item in raw_locations:
        if not isinstance(item, dict):
            continue
        try:
            lat = float(item["lat"])
            lng = float(item["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        role = str(item.get("role", "site")).lower()
        if role not in allowed_roles:
            role = "site"
        locations.append(
            {
                "name": str(item.get("name", "Location"))[:80],
                "lat": lat,
                "lng": lng,
                "caption": str(item.get("caption", ""))[:220],
                "role": role,
            }
        )
    return locations[:8]


def unused_topics(topics: list[dict]) -> list[dict]:
    return [topic for topic in topics if not topic.get("used_on")]


def choose_topic(topics: list[dict]) -> tuple[dict, dict]:
    unused = unused_topics(topics)
    if not unused:
        for topic in topics:
            topic["used_on"] = None
        unused = topics
    current = unused[0]
    next_topic = unused[1] if len(unused) > 1 else topics[0]
    return current, next_topic


def article_prompt(topic: dict, next_topic: dict) -> str:
    return f"""
Write today's historical read for a personal site called The Daily Brief.

Topic: {topic["title"]}
Era: {topic.get("era", "")}
Region: {topic.get("region", "")}
Tomorrow's topic teaser: {next_topic["title"]}

Requirements:
- 2,000 to 2,500 words.
- Vivid narrative hook.
- "The world before" context.
- The event or period itself told as a story.
- The scholarly debate or mystery, if one exists.
- The aftermath and what rose from it.
- A closing "Why it matters" section connecting the story to durable modern themes.
- Narrative-driven, intellectually honest about uncertainty, no listicle tone.
- End with one line: "Further reading: ..." citing 2-3 real books you are confident exist.
- After that, add "Tomorrow's read: {next_topic["title"]}."

Return the article in Markdown.

After the article, include a fenced JSON block in this exact shape:
```json
{{"locations":[{{"name":"...","lat":0.0,"lng":0.0,"caption":"...","role":"capital|battle|city|site"}}]}}
```
Include 3 to 8 key locations for the map. Use decimal latitude and longitude.
""".strip()


def demo_article(topic: dict, next_topic: dict) -> dict:
    title = topic["title"]
    article = f"""
## The archive before dawn

The first edition preview is using local sample content because no Anthropic API key is available in this environment. It is still shaped like a real Daily Brief essay so the page, archive, map, and mobile layout can be reviewed before the repository secret is added.

The story begins with {title}, a reminder that history often survives through accidents of power, weather, conquest, and paperwork. In the ancient Near East, rulers collected words as carefully as they collected tribute. Clay tablets, palace rooms, trained scribes, and imperial messengers formed a knowledge system that could outlast the people who built it.

## The world before

Before modern archives, knowledge was physical in the most literal sense. A record had weight. It had to be pressed into wet clay, dried, shelved, copied, and guarded. Cities competed not only with armies but with calendars, contracts, omens, poems, and lists. The stronger a state became, the more it needed memory: taxes, grain, land, soldiers, borders, treaties, and divine favor all had to be recorded.

Assyria turned that need into an imperial habit. Its kings ruled through speed and fear, but also through information. Reports moved from provincial governors to the palace. Scholars watched the sky and read the liver of sacrificed animals for signs. Scribes copied older stories because the past was not dead material; it was a reservoir of authority.

## The event itself

The library associated with Ashurbanipal at Nineveh was not a public library in the modern sense. It was a royal instrument. Tablets gathered from across Mesopotamia gave the king access to ritual knowledge, lexical lists, medical texts, royal inscriptions, hymns, prayers, omen series, and literature. Among them were versions of the Epic of Gilgamesh, one of the oldest surviving works of world literature.

What makes the library feel almost modern is the ambition behind it. The palace wanted not just a few useful documents but a systematic command of inherited learning. The king presented himself as literate, trained, and uniquely capable of reading difficult texts. Whether that image was fully true matters less than the political message: empire claimed the right to collect the world's memory and arrange it around the throne.

## Debate and uncertainty

Scholars still debate how centralized the collection really was, how much Ashurbanipal personally directed it, and how many tablets belonged to separate palace archives later discovered together. The ancient evidence is rich but fragmentary. The burned destruction layers that helped preserve tablets also shattered them. Modern knowledge depends on painstaking reconstruction, joins between fragments, and comparisons with tablets found elsewhere.

That uncertainty is part of the drama. Archives look stable from a distance, but they are living systems: copied, moved, broken, misfiled, looted, recovered, and reinterpreted. Every generation asks the old clay new questions.

## Aftermath

Nineveh fell in 612 BC, and Assyrian imperial power collapsed. Yet the tablets survived because disaster fired and buried them. The library became a time capsule created unintentionally by conquest. When nineteenth-century excavators uncovered the tablets, they opened a route back into Mesopotamian literature, science, ritual, and administration.

The result changed the study of the ancient world. Stories once known only from later traditions suddenly had deeper roots. Mesopotamia emerged not as a footnote to classical history but as a civilization with its own dense intellectual life.

## Why it matters

The library matters because it shows that power and knowledge have always been tangled together. States collect information to govern, but archives can escape their creators. A royal tool can become a common inheritance. A destroyed palace can become a classroom for the future.

Further reading: Andrew George, The Epic of Gilgamesh; Karen Radner, Ancient Assyria: A Very Short Introduction; Irving Finkel, The Ark Before Noah.

Tomorrow's read: {next_topic["title"]}.
""".strip()
    return {
        "title": title,
        "html": markdown_to_html(article),
        "locations": [
            {
                "name": "Nineveh",
                "lat": 36.3594,
                "lng": 43.1529,
                "caption": "The Assyrian capital where many tablets from Ashurbanipal's library were found.",
                "role": "capital",
            },
            {
                "name": "Ashur",
                "lat": 35.456,
                "lng": 43.262,
                "caption": "Older Assyrian religious and political center on the Tigris.",
                "role": "city",
            },
            {
                "name": "Babylon",
                "lat": 32.5364,
                "lng": 44.4208,
                "caption": "A major Mesopotamian center whose traditions shaped Assyrian scholarship.",
                "role": "site",
            },
        ],
        "badge": "local preview",
        "source": "preview",
        "generated_at": iso_now_utc(),
    }


def generate_article(topic: dict, next_topic: dict) -> dict:
    text, _, usage = anthropic_message(
        prompt=article_prompt(topic, next_topic),
        model=ARTICLE_MODEL,
        max_tokens=7500,
        system="You are a careful narrative historian writing for a curious morning reader.",
        temperature=0.8,
        timeout=240,
    )
    data, article_text = extract_json_block(text)
    locations = []
    if isinstance(data, dict):
        locations = validate_locations(data.get("locations"))
    else:
        print("Article map JSON was missing or malformed; rendering without a map.")
    return {
        "title": topic["title"],
        "html": markdown_to_html(article_text),
        "locations": locations,
        "badge": None,
        "source": "api",
        "usage": usage,
        "generated_at": iso_now_utc(),
    }


def sports_prompt(sport: str, recent_questions: list[str]) -> str:
    recent = "\n".join(f"- {question}" for question in recent_questions[-30:]) or "- none"
    return f"""
Create one Daily Sports Trivia item for The Daily Brief.

Today's sport lane: {sport}
Difficulty mix target: about 60 percent hard, 40 percent medium.
Use one of these formats: straight question, name-the-player career clues, or stat-based puzzle.
Avoid repeating these recent questions:
{recent}

Return only JSON:
{{
  "sport": "{sport}",
  "difficulty": "medium|hard",
  "question": "...",
  "answer": "...",
  "story": "2-3 sentences explaining the story behind the answer."
}}
""".strip()


def demo_trivia(sport: str) -> dict:
    return {
        "sport": sport,
        "difficulty": "hard",
        "question": "Which NFL team won Super Bowl XX with one of the most dominant defenses in league history?",
        "answer": "The 1985 Chicago Bears.",
        "story": "Chicago finished the season 18-1 including the playoffs and routed New England 46-10 in the Super Bowl. Their defense became a cultural artifact as much as a football unit.",
        "badge": "local preview",
        "source": "preview",
        "generated_at": iso_now_utc(),
    }


def generate_trivia(sport: str, history: list[dict]) -> dict:
    recent_questions = [item.get("question", "") for item in history if item.get("question")]
    text, _, usage = anthropic_message(
        prompt=sports_prompt(sport, recent_questions),
        model=TRIVIA_MODEL,
        max_tokens=900,
        system="You write crisp, factually accurate sports trivia.",
        temperature=0.9,
        timeout=90,
    )
    data = parse_json_object(text)
    if not data:
        raise RuntimeError("Trivia response was not valid JSON.")
    for key in ("question", "answer", "story"):
        if not data.get(key):
            raise RuntimeError(f"Trivia response missing {key}.")
    return {
        "sport": str(data.get("sport", sport)),
        "difficulty": str(data.get("difficulty", "medium")),
        "question": str(data["question"]),
        "answer": str(data["answer"]),
        "story": str(data["story"]),
        "badge": None,
        "source": "api",
        "usage": usage,
        "generated_at": iso_now_utc(),
    }


def market_prompt(now: datetime) -> str:
    monday_line = ""
    if now.weekday() == 0:
        monday_line = "Because today is Monday, include one line on major weekend news if web search surfaces anything market-moving."
    return f"""
Generate a concise US Markets Today card for a reader in Chicago.

Current Central Time: {compact_ct_timestamp(now)}
{monday_line}

Use web search for current market information. Prioritize:
1. Where US index futures point pre-open and why.
2. One-line recap of yesterday's close.
3. Today's key economic releases and times, including CPI, jobs data, Fed speakers, or FOMC if relevant.
4. Notable earnings reporting today.
5. Major overnight global market moves.
6. Rates, oil, and dollar only if actually moving.
7. One theme to watch.

No investment advice. No filler. If only six things matter, write six.
Return only JSON:
{{"bullets":["one or two tight sentences","..."]}}
""".strip()


def demo_markets(now: datetime) -> dict:
    return {
        "title": "US Markets Today",
        "bullets": [
            "Local preview mode is active because ANTHROPIC_API_KEY is not set here.",
            "In GitHub Actions, this card will use Claude with web search for futures, yesterday's close, economic data, earnings, global moves, and one theme to watch.",
            "The timestamp below lets you see exactly when the card was generated.",
        ],
        "generated_at_ct": compact_ct_timestamp(now),
        "badge": "local preview",
        "sources": [],
        "source": "preview",
        "generated_at": iso_now_utc(),
    }


def closed_weekend_markets(now: datetime) -> dict:
    return {
        "title": "US Markets Today",
        "bullets": [
            "US equity markets are closed for the weekend.",
            "The next fresh market briefing will be generated on the next weekday run.",
        ],
        "generated_at_ct": compact_ct_timestamp(now),
        "badge": "weekend",
        "sources": [],
        "source": "static",
        "generated_at": iso_now_utc(),
    }


def generate_markets(now: datetime) -> dict:
    if os.getenv("SIMULATE_MARKETS_FAILURE") == "1":
        raise RuntimeError("SIMULATE_MARKETS_FAILURE=1 requested.")
    text, citations, usage = anthropic_message(
        prompt=market_prompt(now),
        model=MARKETS_MODEL,
        max_tokens=1400,
        system="You are a concise pre-market briefing writer. Be factual, current, and careful.",
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 6,
                "user_location": {
                    "type": "approximate",
                    "city": "Chicago",
                    "region": "Illinois",
                    "country": "US",
                    "timezone": "America/Chicago",
                },
            }
        ],
        temperature=0.3,
        timeout=180,
    )
    data = parse_json_object(text)
    if data and isinstance(data.get("bullets"), list):
        bullets = [str(item).strip() for item in data["bullets"] if str(item).strip()]
    else:
        bullets = [
            re.sub(r"^[-*\d.\s]+", "", line).strip()
            for line in text.splitlines()
            if re.sub(r"^[-*\d.\s]+", "", line).strip()
        ]
    if not bullets:
        raise RuntimeError("Markets response did not contain bullets.")
    return {
        "title": "US Markets Today",
        "bullets": bullets[:10],
        "generated_at_ct": compact_ct_timestamp(now),
        "badge": None,
        "sources": citations[:4],
        "source": "api",
        "usage": usage,
        "generated_at": iso_now_utc(),
    }


def topics_prompt(existing_titles: list[str]) -> str:
    existing = "\n".join(f"- {title}" for title in existing_titles)
    return f"""
The Daily Brief needs 30 fresh historical read topics.

Existing topics to avoid:
{existing}

Return only JSON as a list of 30 objects:
[{{"title":"...","era":"Ancient|Medieval|Early Modern|Modern","region":"..."}}]

Rules:
- No duplicates or near duplicates.
- Span all continents and many time periods from 2000 BC to the 20th century.
- Prefer events, periods, cities, movements, and turning points with narrative depth.
""".strip()


def append_fresh_topics(topics: list[dict]) -> bool:
    existing_titles = [topic.get("title", "") for topic in topics]
    text, _, _ = anthropic_message(
        prompt=topics_prompt(existing_titles),
        model=TOPICS_MODEL,
        max_tokens=2500,
        system="You curate diverse, historically meaningful article topics.",
        temperature=0.8,
        timeout=120,
    )
    data, _ = extract_json_block(text)
    if data is None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Topic refresh returned invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError("Topic refresh did not return a list.")
    existing_norms = {re.sub(r"\W+", "", title.lower()) for title in existing_titles}
    added = 0
    for item in data:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        norm = re.sub(r"\W+", "", str(item["title"]).lower())
        if norm in existing_norms:
            continue
        topics.append(
            {
                "title": str(item["title"]),
                "era": str(item.get("era", "")),
                "region": str(item.get("region", "")),
                "used_on": None,
            }
        )
        existing_norms.add(norm)
        added += 1
    print(f"Added {added} fresh topics.")
    return added > 0


def reuse_previous(state: dict, key: str, badge: str, reason: str) -> dict | None:
    previous = copy.deepcopy(state.get("last_successful", {}).get(key))
    if not previous:
        return None
    previous["badge"] = badge
    previous["reused_previous"] = True
    previous["reuse_reason"] = reason
    return previous


def render_badge(label: str | None) -> str:
    if not label:
        return ""
    return f'<span class="badge">{html.escape(label)}</span>'


def render_map(article: dict) -> str:
    locations = article.get("locations") or []
    if not locations:
        return ""
    payload = html.escape(json.dumps(locations, ensure_ascii=False), quote=True)
    return f"""
      <div class="map-wrap">
        <div id="history-map" data-locations="{payload}" aria-label="Map of key locations"></div>
      </div>
""".rstrip()


def render_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    links = []
    for item in sources:
        title = html.escape(str(item.get("title", "Source")))
        url = html.escape(str(item.get("url", "#")), quote=True)
        links.append(f'<a href="{url}" rel="noopener noreferrer">{title}</a>')
    return '<p class="sources">Sources: ' + " · ".join(links) + "</p>"


def render_page(
    *,
    page_date: datetime,
    edition: int,
    markets: dict,
    trivia: dict,
    article: dict,
    ticker_path: str,
    archive_prefix: str = "archive/",
) -> str:
    map_html = render_map(article)
    markets_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in markets.get("bullets", []))
    sources_html = render_sources(markets.get("sources", []))
    article_html = article.get("html", "")
    trivia_answer = html.escape(str(trivia.get("answer", "")))
    trivia_story = html.escape(str(trivia.get("story", "")))
    trivia_question = html.escape(str(trivia.get("question", "")))
    ticker_path_json = json.dumps(ticker_path)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Daily Brief - {html.escape(page_date.date().isoformat())}</title>
  <link rel="preconnect" href="https://unpkg.com">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111315;
      --panel: #1b2025;
      --panel-2: #202832;
      --text: #f2efe8;
      --muted: #aeb7bd;
      --line: #303943;
      --gold: #e6b450;
      --green: #6fca93;
      --blue: #86b7ff;
      --red: #ef7d7d;
      --shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      padding: 0 0 4.75rem;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.62;
    }}
    a {{ color: var(--blue); }}
    .page {{
      width: min(100% - 28px, 700px);
      margin: 0 auto;
      padding: 1.15rem 0 3rem;
    }}
    header {{
      padding: 1.1rem 0 1.2rem;
      border-bottom: 1px solid var(--line);
      margin-bottom: 1rem;
    }}
    .kicker {{
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    h1, h2, h3 {{ line-height: 1.16; letter-spacing: 0; }}
    h1 {{
      margin: 0.15rem 0 0.25rem;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.15rem, 12vw, 4rem);
      font-weight: 700;
    }}
    .dateline {{
      color: var(--muted);
      font-size: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      align-items: center;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin: 1rem 0;
      padding: 1rem;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 1.22rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 1.55rem;
      border-radius: 999px;
      border: 1px solid rgba(230, 180, 80, 0.45);
      color: var(--gold);
      padding: 0.1rem 0.55rem;
      font-size: 0.76rem;
      white-space: nowrap;
    }}
    .markets ul {{
      margin: 0;
      padding-left: 1.1rem;
    }}
    .markets li {{ margin: 0.45rem 0; }}
    .stamp, .sources {{
      margin: 0.75rem 0 0;
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .sources a {{ color: var(--muted); }}
    .trivia-meta {{
      color: var(--green);
      font-size: 0.86rem;
      margin-bottom: 0.35rem;
    }}
    .question {{
      margin: 0.35rem 0 0.9rem;
      font-size: 1.06rem;
      font-weight: 650;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 8px;
      background: var(--gold);
      color: #1b1403;
      font: inherit;
      font-weight: 750;
      min-height: 44px;
      padding: 0.62rem 0.9rem;
      cursor: pointer;
    }}
    button:focus-visible {{
      outline: 3px solid var(--blue);
      outline-offset: 3px;
    }}
    .answer {{
      display: none;
      margin-top: 0.9rem;
      padding-top: 0.9rem;
      border-top: 1px solid var(--line);
    }}
    .answer.is-visible {{ display: block; }}
    .answer strong {{ color: var(--gold); }}
    .article-shell {{
      margin-top: 1.45rem;
      padding-top: 0.7rem;
      border-top: 1px solid var(--line);
    }}
    .article-title {{
      margin: 0.4rem 0 0.8rem;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 9vw, 3.2rem);
    }}
    .map-wrap {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0.9rem 0 1.25rem;
      background: var(--panel-2);
    }}
    #history-map {{
      width: 100%;
      height: min(68vh, 420px);
      min-height: 280px;
    }}
    article {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.1rem;
      line-height: 1.74;
    }}
    article h2 {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 1.35rem;
      margin: 1.7rem 0 0.45rem;
    }}
    article h3 {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 1.08rem;
      color: var(--gold);
      margin: 1.3rem 0 0.4rem;
    }}
    article p {{ margin: 0 0 1rem; }}
    .further-reading {{
      border-left: 3px solid var(--gold);
      padding-left: 0.8rem;
      color: #e8dfcc;
    }}
    footer {{
      border-top: 1px solid var(--line);
      margin-top: 2rem;
      padding-top: 1rem;
      color: var(--muted);
    }}
    .ticker {{
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 1000;
      background: #060708;
      border-top: 1px solid var(--line);
      min-height: 3.5rem;
      display: flex;
      align-items: center;
      overflow: hidden;
      touch-action: pan-y;
    }}
    .ticker-label {{
      flex: 0 0 auto;
      align-self: stretch;
      display: flex;
      align-items: center;
      padding: 0 0.75rem;
      background: var(--red);
      color: #1b0505;
      font-weight: 850;
      font-size: 0.78rem;
    }}
    .ticker-window {{ overflow: hidden; flex: 1; }}
    .ticker-track {{
      display: inline-flex;
      gap: 2rem;
      white-space: nowrap;
      animation: marquee 55s linear infinite;
      will-change: transform;
      padding-left: 100%;
    }}
    .ticker.paused .ticker-track,
    .ticker:hover .ticker-track {{ animation-play-state: paused; }}
    .ticker-item {{
      display: inline-flex;
      align-items: center;
      min-height: 3.5rem;
      color: var(--text);
      font-size: 0.95rem;
    }}
    @keyframes marquee {{
      from {{ transform: translateX(0); }}
      to {{ transform: translateX(-100%); }}
    }}
    @media (max-width: 430px) {{
      .page {{ width: min(100% - 22px, 700px); }}
      .card {{ padding: 0.9rem; }}
      .section-head {{ align-items: flex-start; }}
      article {{ font-size: 1.06rem; }}
      .ticker-label {{ padding: 0 0.55rem; }}
      .ticker-track {{ gap: 1.4rem; animation-duration: 48s; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div class="kicker">Personal morning edition</div>
      <h1>The Daily Brief</h1>
      <div class="dateline">
        <span>{html.escape(human_date(page_date))}</span>
        <span>Edition #{edition}</span>
      </div>
    </header>

    <section class="card markets" aria-labelledby="markets-heading">
      <div class="section-head">
        <h2 id="markets-heading">US Markets Today</h2>
        {render_badge(markets.get("badge"))}
      </div>
      <ul>
        {markets_items}
      </ul>
      <p class="stamp">Generated {html.escape(str(markets.get("generated_at_ct", "")))}</p>
      {sources_html}
    </section>

    <section class="card trivia" aria-labelledby="trivia-heading">
      <div class="section-head">
        <h2 id="trivia-heading">Daily Sports Trivia</h2>
        {render_badge(trivia.get("badge"))}
      </div>
      <div class="trivia-meta">{html.escape(str(trivia.get("sport", "")))} · {html.escape(str(trivia.get("difficulty", "")).title())}</div>
      <p class="question">{trivia_question}</p>
      <button id="reveal-answer" type="button" aria-expanded="false" aria-controls="trivia-answer">Reveal Answer</button>
      <div id="trivia-answer" class="answer">
        <p><strong>{trivia_answer}</strong></p>
        <p>{trivia_story}</p>
      </div>
    </section>

    <section class="article-shell" aria-labelledby="article-heading">
      <div class="section-head">
        <div>
          <div class="kicker">Today's Historical Read</div>
          <h2 id="article-heading" class="article-title">{html.escape(str(article.get("title", "")))}</h2>
        </div>
        {render_badge(article.get("badge"))}
      </div>
      {map_html}
      <article>
        {article_html}
      </article>
    </section>

    <footer>
      <a href="{archive_prefix}">Archive</a>
    </footer>
  </main>

  <div class="ticker" id="ticker" aria-label="Current world headlines">
    <div class="ticker-label">HEADLINES</div>
    <div class="ticker-window">
      <div class="ticker-track" id="ticker-track">
        <span class="ticker-item">Loading headlines...</span>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const tickerUrl = {ticker_path_json};
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({{
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }}[char]));

    const answerButton = document.getElementById('reveal-answer');
    const answer = document.getElementById('trivia-answer');
    answerButton?.addEventListener('click', () => {{
      const visible = answer.classList.toggle('is-visible');
      answerButton.setAttribute('aria-expanded', String(visible));
      answerButton.textContent = visible ? 'Hide Answer' : 'Reveal Answer';
    }});

    const ticker = document.getElementById('ticker');
    const track = document.getElementById('ticker-track');
    const setTicker = (items) => {{
      const safeItems = items.length ? items : [{{ source: 'Brief', text: 'Headlines will update shortly.' }}];
      const rendered = safeItems.map((item) =>
        `<span class="ticker-item">[${{escapeHtml(item.source)}}] ${{escapeHtml(item.text)}}</span>`
      ).join('');
      track.innerHTML = rendered + rendered;
    }};
    fetch(tickerUrl, {{ cache: 'no-store' }})
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Ticker fetch failed')))
      .then((data) => setTicker(data.headlines || []))
      .catch(() => setTicker([]));
    ['touchstart', 'pointerdown', 'focusin'].forEach((eventName) => {{
      ticker.addEventListener(eventName, () => ticker.classList.add('paused'));
    }});
    ['touchend', 'pointerup', 'focusout'].forEach((eventName) => {{
      ticker.addEventListener(eventName, () => ticker.classList.remove('paused'));
    }});

    const mapEl = document.getElementById('history-map');
    if (mapEl && window.L) {{
      try {{
        const locations = JSON.parse(mapEl.dataset.locations || '[]');
        if (locations.length) {{
          const map = L.map(mapEl, {{ scrollWheelZoom: false }});
          L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
          }}).addTo(map);
          const colors = {{
            capital: '#e6b450',
            battle: '#ef7d7d',
            city: '#86b7ff',
            site: '#6fca93'
          }};
          const bounds = [];
          locations.forEach((loc) => {{
            const marker = L.circleMarker([loc.lat, loc.lng], {{
              radius: 8,
              color: colors[loc.role] || colors.site,
              fillColor: colors[loc.role] || colors.site,
              fillOpacity: 0.9,
              weight: 2
            }}).addTo(map);
            marker.bindPopup(`<strong>${{escapeHtml(loc.name)}}</strong><br>${{escapeHtml(loc.caption || '')}}`);
            bounds.push([loc.lat, loc.lng]);
          }});
          map.fitBounds(bounds, {{ padding: [28, 28], maxZoom: 6 }});
        }}
      }} catch (error) {{
        mapEl.style.display = 'none';
      }}
    }}
  </script>
</body>
</html>
"""


def render_archive_index(entries: list[tuple[str, str]], generated_at: datetime) -> str:
    links = "\n".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(label)}</a></li>' for filename, label in entries
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Daily Brief Archive</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      background: #111315;
      color: #f2efe8;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
    }}
    main {{
      width: min(100% - 28px, 700px);
      margin: 0 auto;
      padding: 2rem 0;
    }}
    h1 {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.2rem, 11vw, 3.8rem);
      line-height: 1.1;
      margin: 0 0 1rem;
    }}
    a {{ color: #86b7ff; }}
    ul {{ padding-left: 1.1rem; }}
    li {{ margin: 0.5rem 0; }}
    .muted {{ color: #aeb7bd; }}
  </style>
</head>
<body>
  <main>
    <p><a href="../">Back to today's brief</a></p>
    <h1>Archive</h1>
    <p class="muted">Generated {html.escape(compact_ct_timestamp(generated_at))}</p>
    <ul>
      {links or '<li>No archived editions yet.</li>'}
    </ul>
  </main>
</body>
</html>
"""


def archive_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for path in sorted(ARCHIVE_DIR.glob("*.html"), reverse=True):
        if path.name == "index.html":
            continue
        label = path.stem
        try:
            parsed = datetime.strptime(path.stem, "%Y-%m-%d")
            label = human_date(parsed.replace(tzinfo=CENTRAL))
        except ValueError:
            pass
        entries.append((path.name, label))
    return entries


def update_last_successful(state: dict, key: str, section: dict) -> None:
    if section.get("reused_previous"):
        return
    state.setdefault("last_successful", {})[key] = copy.deepcopy(section)


def main() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    now = today_central()
    today_key = now.date().isoformat()
    state = load_json(STATE_FILE, {"edition": 0, "last_successful": {}})
    topics = load_json(TOPICS_FILE, [])
    trivia_history = load_json(TRIVIA_FILE, [])

    if not topics:
        raise RuntimeError("topics.json has no topics.")

    edition = int(state.get("edition", 0)) + 1
    topic, next_topic = choose_topic(topics)
    sport = SPORT_ROTATION[(edition - 1) % len(SPORT_ROTATION)]

    print(f"Building edition #{edition} for {today_key}.")

    try:
        article = generate_article(topic, next_topic)
        topic["used_on"] = today_key
        print(f"Generated article: {topic['title']}")
    except Exception as exc:
        print(f"Article generation failed: {exc}")
        article = reuse_previous(state, "article", "yesterday's edition", str(exc)) or demo_article(topic, next_topic)

    try:
        if now.weekday() >= 5:
            markets = reuse_previous(state, "markets", "Friday's card", "Weekend run")
            markets = markets or closed_weekend_markets(now)
        else:
            markets = generate_markets(now)
        print("Generated markets card.")
    except Exception as exc:
        print(f"Markets generation failed: {exc}")
        markets = reuse_previous(state, "markets", "yesterday's edition", str(exc)) or demo_markets(now)

    try:
        trivia = generate_trivia(sport, trivia_history)
        trivia_history.append(
            {
                "date": today_key,
                "sport": trivia.get("sport"),
                "question": trivia.get("question"),
                "answer": trivia.get("answer"),
            }
        )
        trivia_history = trivia_history[-90:]
        print("Generated sports trivia.")
    except Exception as exc:
        print(f"Trivia generation failed: {exc}")
        trivia = reuse_previous(state, "trivia", "yesterday's edition", str(exc)) or demo_trivia(sport)

    try:
        update_ticker()
    except Exception as exc:
        print(f"Ticker update failed: {exc}. Continuing with previous ticker if available.")

    page = render_page(
        page_date=now,
        edition=edition,
        markets=markets,
        trivia=trivia,
        article=article,
        ticker_path="ticker.json",
    )
    index_path = SITE_DIR / "index.html"
    archive_path = ARCHIVE_DIR / f"{today_key}.html"
    index_path.write_text(page, encoding="utf-8")
    archive_path.write_text(
        render_page(
            page_date=now,
            edition=edition,
            markets=markets,
            trivia=trivia,
            article=article,
            ticker_path="../ticker.json",
            archive_prefix="./",
        ),
        encoding="utf-8",
    )
    (ARCHIVE_DIR / "index.html").write_text(
        render_archive_index(archive_entries(), now),
        encoding="utf-8",
    )

    state["edition"] = edition
    state["updated_at"] = iso_now_utc()
    update_last_successful(state, "article", article)
    update_last_successful(state, "markets", markets)
    update_last_successful(state, "trivia", trivia)

    if len(unused_topics(topics)) < 10:
        try:
            if append_fresh_topics(topics):
                print("Topic queue refreshed.")
        except Exception as exc:
            print(f"Topic refresh failed: {exc}")

    write_json(STATE_FILE, state)
    write_json(TOPICS_FILE, topics)
    write_json(TRIVIA_FILE, trivia_history)
    print(f"Wrote {index_path.relative_to(ROOT)} and archive page {archive_path.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
