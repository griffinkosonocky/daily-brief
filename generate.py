#!/usr/bin/env python3
"""Build The Daily Brief as static files for GitHub Pages."""

from __future__ import annotations

import copy
import html
import json
import os
import random
import re
import sys
import urllib.error
import urllib.parse
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
OPEN_TRIVIA_URL = "https://opentdb.com/api.php"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

ARTICLE_MODEL = os.getenv("ANTHROPIC_ARTICLE_MODEL", "claude-sonnet-4-6")
MARKETS_MODEL = os.getenv("ANTHROPIC_MARKETS_MODEL", "claude-sonnet-4-6")
TOPICS_MODEL = os.getenv("ANTHROPIC_TOPICS_MODEL", "claude-haiku-4-5")

SPORTS_WATCHLIST = [
    {"key": "soccer_fifa_world_cup", "label": "World Cup", "full_name": "FIFA World Cup"},
    {"key": "americanfootball_nfl", "label": "NFL", "full_name": "NFL"},
    {"key": "americanfootball_ncaaf", "label": "CFB", "full_name": "College Football"},
    {"key": "basketball_ncaab", "label": "NCAAB", "full_name": "College Basketball"},
    {"key": "basketball_nba", "label": "NBA", "full_name": "NBA"},
]

BOOKMAKER_PRIORITY = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "espnbet",
    "betrivers",
    "pointsbetus",
    "bovada",
    "betonlineag",
]


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


def fetch_json(url: str, *, timeout: int = 30) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "accept": "application/json",
            "user-agent": "daily-brief/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc
    return json.loads(body), headers


def build_url(base: str, params: dict[str, Any]) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


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


def normalize_key(value: str) -> str:
    return re.sub(r"\W+", "", value.lower())


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


def extract_trailing_json(text: str) -> tuple[dict | list | None, str]:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(stripped) if char in "{["]
    for index in reversed(starts):
        candidate = stripped[index:].strip()
        try:
            data, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[end:].strip():
            continue
        if isinstance(data, (dict, list)):
            return data, stripped[:index].strip()
    return None, stripped


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
    data, cleaned = extract_trailing_json(text)
    if data is not None:
        return data, cleaned
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


def demo_trivia() -> dict:
    return {
        "sport": "Sports",
        "difficulty": "hard",
        "question": "Which NFL team won Super Bowl XX with one of the most dominant defenses in league history?",
        "answer": "The 1985 Chicago Bears.",
        "story": "Chicago finished 18-1 including the postseason and beat New England 46-10 in Super Bowl XX. That defense became the reference point for modern NFL dominance.",
        "options": [
            "1985 Chicago Bears",
            "2000 Baltimore Ravens",
            "1976 Pittsburgh Steelers",
            "2013 Seattle Seahawks",
        ],
        "badge": "local preview",
        "source": "preview",
        "generated_at": iso_now_utc(),
    }


def generate_trivia(history: list[dict]) -> dict:
    recent_questions = [item.get("question", "") for item in history if item.get("question")]
    recent_keys = {normalize_key(str(question)) for question in recent_questions[-30:]}

    url = build_url(
        OPEN_TRIVIA_URL,
        {
            "amount": 10,
            "category": 21,
            "type": "multiple",
        },
    )
    data, _ = fetch_json(url, timeout=30)
    if not isinstance(data, dict):
        raise RuntimeError("Trivia API returned an unexpected response.")
    response_code = data.get("response_code")
    if response_code != 0:
        raise RuntimeError(f"Trivia API response code {response_code}.")

    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("Trivia API returned no questions.")

    selected = None
    for item in results:
        question = html.unescape(str(item.get("question", ""))).strip()
        if question and normalize_key(question) not in recent_keys:
            selected = item
            break
    selected = selected or results[0]

    question = html.unescape(str(selected.get("question", ""))).strip()
    answer = html.unescape(str(selected.get("correct_answer", ""))).strip()
    incorrect = [
        html.unescape(str(option)).strip()
        for option in selected.get("incorrect_answers", [])
        if str(option).strip()
    ]
    if not question or not answer:
        raise RuntimeError("Trivia API question was incomplete.")

    options = incorrect + [answer]
    random.shuffle(options)
    category = html.unescape(str(selected.get("category", "Sports"))).replace("Entertainment: ", "")
    difficulty = str(selected.get("difficulty", "medium")).lower()
    return {
        "sport": category,
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "story": "Pulled from Open Trivia DB's Sports category, with no model-written facts.",
        "options": options,
        "badge": None,
        "source": "opentdb",
        "generated_at": iso_now_utc(),
    }


def remember_trivia(history: list[dict], today_key: str, trivia: dict) -> list[dict]:
    question = str(trivia.get("question", "")).strip()
    answer = str(trivia.get("answer", "")).strip()
    if not question or not answer:
        return history[-90:]

    question_key = normalize_key(question)
    for item in history:
        if normalize_key(str(item.get("question", ""))) == question_key:
            item["date"] = today_key
            item["sport"] = trivia.get("sport")
            item["answer"] = answer
            return history[-90:]

    history.append(
        {
            "date": today_key,
            "sport": trivia.get("sport"),
            "question": question,
            "answer": answer,
        }
    )
    return history[-90:]


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_time_ct(value: datetime | None) -> str:
    if not value:
        return "TBD"
    local = value.astimezone(CENTRAL)
    hour = local.strftime("%I").lstrip("0") or "0"
    return local.strftime(f"{hour}:%M %p CT")


def is_today_ct(value: datetime | None, now: datetime) -> bool:
    return bool(value and value.astimezone(CENTRAL).date() == now.date())


def day_bounds_utc(now: datetime) -> tuple[str, str]:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return iso_utc_z(start), iso_utc_z(end)


def odds_api_url(sport_key: str, endpoint: str, params: dict[str, Any]) -> str:
    path = f"{ODDS_API_BASE}/sports/{urllib.parse.quote(sport_key)}/{endpoint.strip('/')}/"
    return build_url(path, params)


def format_american_price(value: Any) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return ""
    return f"+{number}" if number > 0 else str(number)


def format_spread_point(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(number) < 0.05:
        return "PK"
    formatted = f"{number:g}"
    return f"+{formatted}" if number > 0 else formatted


def score_lookup(scores: Any) -> dict[str, str]:
    if not isinstance(scores, list):
        return {}
    lookup: dict[str, str] = {}
    for score in scores:
        if not isinstance(score, dict):
            continue
        name = str(score.get("name", "")).strip()
        value = str(score.get("score", "")).strip()
        if name and value:
            lookup[name] = value
    return lookup


def bookmaker_sort_key(bookmaker: dict) -> tuple[int, str]:
    key = str(bookmaker.get("key", ""))
    try:
        rank = BOOKMAKER_PRIORITY.index(key)
    except ValueError:
        rank = len(BOOKMAKER_PRIORITY)
    return rank, str(bookmaker.get("title", key))


def extract_spread(odds_item: dict) -> dict | None:
    bookmakers = odds_item.get("bookmakers")
    if not isinstance(bookmakers, list):
        return None

    for bookmaker in sorted((item for item in bookmakers if isinstance(item, dict)), key=bookmaker_sort_key):
        markets = bookmaker.get("markets")
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "spreads":
                continue
            outcomes = []
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name", "")).strip()
                point_label = format_spread_point(outcome.get("point"))
                price_label = format_american_price(outcome.get("price"))
                if not name or not point_label:
                    continue
                outcomes.append(
                    {
                        "name": name,
                        "point": outcome.get("point"),
                        "point_label": point_label,
                        "price": outcome.get("price"),
                        "price_label": price_label,
                    }
                )
            if not outcomes:
                continue
            favorite = next(
                (
                    outcome
                    for outcome in outcomes
                    if isinstance(outcome.get("point"), (int, float)) and float(outcome["point"]) < 0
                ),
                outcomes[0],
            )
            last_update = parse_iso_datetime(bookmaker.get("last_update"))
            return {
                "book": str(bookmaker.get("title", bookmaker.get("key", "Sportsbook"))),
                "last_update_ct": compact_time_ct(last_update) if last_update else "",
                "summary": f"{favorite['name']} {favorite['point_label']}",
                "outcomes": outcomes,
            }
    return None


def normalize_game(score_item: dict | None, odds_item: dict | None, sport: dict, now: datetime) -> dict:
    source = score_item or odds_item or {}
    commence_at = parse_iso_datetime(source.get("commence_time"))
    scores = score_lookup((score_item or {}).get("scores"))
    home_team = str(source.get("home_team", "Home")).strip()
    away_team = str(source.get("away_team", "Away")).strip()
    home_score = scores.get(home_team, "")
    away_score = scores.get(away_team, "")
    completed = bool((score_item or {}).get("completed"))
    has_score = bool(home_score or away_score)

    status_key = "upcoming"
    status_label = compact_time_ct(commence_at)
    if completed:
        status_key = "final"
        status_label = "Final"
    elif has_score:
        status_key = "live"
        status_label = "Live"
    elif commence_at and commence_at <= now.astimezone(timezone.utc):
        status_key = "live"
        status_label = "In progress"

    return {
        "id": str(source.get("id", "")),
        "sport_key": sport["key"],
        "league": sport["label"],
        "league_name": sport["full_name"],
        "status_key": status_key,
        "status_label": status_label,
        "commence_time": source.get("commence_time"),
        "commence_ct": compact_time_ct(commence_at),
        "sort_time": commence_at.isoformat() if commence_at else "",
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "spread": extract_spread(odds_item or {}),
    }


def auth_error(message: str) -> bool:
    return "HTTP 401" in message or "HTTP 403" in message


def collect_sport_games(sport: dict, api_key: str, now: datetime) -> tuple[dict, list[dict[str, str]]]:
    headers_seen: list[dict[str, str]] = []
    scores: list[dict] = []
    odds: list[dict] = []
    errors: list[str] = []

    try:
        score_data, headers = fetch_json(
            odds_api_url(
                sport["key"],
                "scores",
                {"apiKey": api_key, "daysFrom": 1, "dateFormat": "iso"},
            ),
            timeout=30,
        )
        headers_seen.append(headers)
        if isinstance(score_data, list):
            scores = [item for item in score_data if isinstance(item, dict)]
    except RuntimeError as exc:
        message = str(exc)
        if auth_error(message):
            raise
        errors.append(message)

    commence_from, commence_to = day_bounds_utc(now)
    try:
        odds_data, headers = fetch_json(
            odds_api_url(
                sport["key"],
                "odds",
                {
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "spreads",
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                    "commenceTimeFrom": commence_from,
                    "commenceTimeTo": commence_to,
                },
            ),
            timeout=30,
        )
        headers_seen.append(headers)
        if isinstance(odds_data, list):
            odds = [item for item in odds_data if isinstance(item, dict)]
    except RuntimeError as exc:
        message = str(exc)
        if auth_error(message):
            raise
        errors.append(message)

    score_by_id = {
        str(item.get("id", "")): item
        for item in scores
        if item.get("id") and is_today_ct(parse_iso_datetime(item.get("commence_time")), now)
    }
    odds_by_id = {
        str(item.get("id", "")): item
        for item in odds
        if item.get("id") and is_today_ct(parse_iso_datetime(item.get("commence_time")), now)
    }
    game_ids = sorted(set(score_by_id) | set(odds_by_id))
    games = [normalize_game(score_by_id.get(game_id), odds_by_id.get(game_id), sport, now) for game_id in game_ids]
    games.sort(key=lambda game: game.get("sort_time") or "9999")

    note = ""
    if not games:
        note = "No games today."
        if errors:
            note = "Feed unavailable for this sport."

    return (
        {
            "key": sport["key"],
            "label": sport["label"],
            "full_name": sport["full_name"],
            "games": games,
            "note": note,
            "errors": errors[:2],
        },
        headers_seen,
    )


def summarize_sports(sports: list[dict]) -> dict:
    games = [game for sport in sports for game in sport.get("games", [])]
    return {
        "games": len(games),
        "live": sum(1 for game in games if game.get("status_key") == "live"),
        "final": sum(1 for game in games if game.get("status_key") == "final"),
        "upcoming": sum(1 for game in games if game.get("status_key") == "upcoming"),
        "spreads": sum(1 for game in games if game.get("spread")),
    }


def summarize_quota(headers_seen: list[dict[str, str]]) -> dict:
    quota: dict[str, str] = {}
    for headers in headers_seen:
        for source, target in (
            ("x-requests-remaining", "remaining"),
            ("x-requests-used", "used"),
            ("x-requests-last", "last_request"),
        ):
            if headers.get(source):
                quota[target] = headers[source]
    return quota


def demo_sports(now: datetime, reason: str) -> dict:
    sports = [
        {
            "key": sport["key"],
            "label": sport["label"],
            "full_name": sport["full_name"],
            "games": [],
            "note": "Live feed pending.",
            "errors": [],
        }
        for sport in SPORTS_WATCHLIST
    ]
    return {
        "title": "Today's Sports Slate",
        "date_key": now.date().isoformat(),
        "generated_at_ct": compact_ct_timestamp(now),
        "sports": sports,
        "summary": summarize_sports(sports),
        "quota": {},
        "badge": "setup needed",
        "source": "preview",
        "error_reason": reason,
        "generated_at": iso_now_utc(),
    }


def generate_sports_dashboard(now: datetime) -> dict:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not set.")

    sports: list[dict] = []
    headers_seen: list[dict[str, str]] = []
    for sport in SPORTS_WATCHLIST:
        sport_games, sport_headers = collect_sport_games(sport, api_key, now)
        sports.append(sport_games)
        headers_seen.extend(sport_headers)

    return {
        "title": "Today's Sports Slate",
        "date_key": now.date().isoformat(),
        "generated_at_ct": compact_ct_timestamp(now),
        "sports": sports,
        "summary": summarize_sports(sports),
        "quota": summarize_quota(headers_seen),
        "badge": None,
        "source": "odds-api",
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


def render_location_fallback(locations: list[dict]) -> str:
    items = []
    for location in locations:
        role = str(location.get("role", "site"))
        if role not in {"capital", "battle", "city", "site"}:
            role = "site"
        caption = str(location.get("caption", "")).strip()
        caption_html = f'<div class="map-caption">{html.escape(caption)}</div>' if caption else ""
        items.append(
            '<li>'
            f'<span class="map-dot map-dot-{html.escape(role)}" aria-hidden="true"></span>'
            "<div>"
            f'<strong>{html.escape(str(location.get("name", "Location")))}</strong>'
            f"{caption_html}"
            "</div>"
            "</li>"
        )
    return (
        '<div class="map-fallback" id="history-map-fallback">'
        '<div class="map-fallback-title">Key locations</div>'
        f'<ul>{"".join(items)}</ul>'
        "</div>"
    )


def render_map(article: dict) -> str:
    locations = article.get("locations") or []
    if not locations:
        return ""
    payload = html.escape(json.dumps(locations, ensure_ascii=False), quote=True)
    fallback = render_location_fallback(locations)
    return f"""
      <div class="map-wrap" data-map-wrap>
        <div id="history-map" data-locations="{payload}" aria-label="Map of key locations"></div>
        {fallback}
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


def render_trivia_options(trivia: dict) -> str:
    options = trivia.get("options")
    if not isinstance(options, list) or not options:
        return ""
    items = "".join(f"<li>{html.escape(str(option))}</li>" for option in options[:6])
    return f'<ul class="trivia-options">{items}</ul>'


def render_sports_summary(sports: dict) -> str:
    summary = sports.get("summary") if isinstance(sports.get("summary"), dict) else {}
    items = [
        ("Games", summary.get("games", 0)),
        ("Live", summary.get("live", 0)),
        ("Final", summary.get("final", 0)),
        ("Spreads", summary.get("spreads", 0)),
    ]
    return "".join(
        '<div class="sports-stat">'
        f'<span>{html.escape(label)}</span>'
        f"<strong>{html.escape(str(value))}</strong>"
        "</div>"
        for label, value in items
    )


def all_sports_games(sports: dict) -> list[dict]:
    sections = sports.get("sports")
    if not isinstance(sections, list):
        return []
    games = [game for section in sections for game in section.get("games", []) if isinstance(game, dict)]
    games.sort(key=lambda game: game.get("sort_time") or "9999")
    return games


def render_spread(spread: dict | None) -> str:
    if not spread:
        return (
            '<div class="spread-box spread-empty">'
            "<span>Spread</span>"
            "<strong>No line posted</strong>"
            "</div>"
        )
    outcomes = []
    for outcome in spread.get("outcomes", [])[:2]:
        price = outcome.get("price_label")
        price_html = f' <span class="spread-price">{html.escape(str(price))}</span>' if price else ""
        outcomes.append(
            '<div class="spread-line">'
            f'<span>{html.escape(str(outcome.get("name", "")))}</span>'
            f'<strong>{html.escape(str(outcome.get("point_label", "")))}{price_html}</strong>'
            "</div>"
        )
    meta = html.escape(str(spread.get("book", "")))
    if spread.get("last_update_ct"):
        meta += f" · {html.escape(str(spread.get('last_update_ct')))}"
    return (
        '<div class="spread-box">'
        "<span>Spread</span>"
        f"<strong>{html.escape(str(spread.get('summary', 'Line posted')))}</strong>"
        f"{''.join(outcomes)}"
        f'<small>{meta}</small>'
        "</div>"
    )


def render_game_card(game: dict) -> str:
    status_key = html.escape(str(game.get("status_key", "upcoming")))
    away_score = str(game.get("away_score", "")).strip()
    home_score = str(game.get("home_score", "")).strip()
    scoreless = " is-scoreless" if not away_score and not home_score else ""
    return (
        f'<article class="game-card{scoreless}" data-game-status="{status_key}">'
        '<div class="game-meta">'
        f'<span class="league-tag">{html.escape(str(game.get("league", "")))}</span>'
        f'<span class="game-status status-{status_key}">{html.escape(str(game.get("status_label", "")))}</span>'
        "</div>"
        '<div class="teams">'
        '<div class="team-row">'
        f'<span>{html.escape(str(game.get("away_team", "")))}</span>'
        f'<strong>{html.escape(away_score)}</strong>'
        "</div>"
        '<div class="team-row home-team">'
        f'<span>{html.escape(str(game.get("home_team", "")))}</span>'
        f'<strong>{html.escape(home_score)}</strong>'
        "</div>"
        "</div>"
        f'{render_spread(game.get("spread"))}'
        "</article>"
    )


def render_sports_filters(sports: dict) -> str:
    sections = sports.get("sports") if isinstance(sports.get("sports"), list) else []
    buttons = [
        '<button class="league-filter is-active" type="button" data-sport-filter="all">All</button>'
    ]
    for section in sections:
        label = str(section.get("label", "")).strip()
        if not label:
            continue
        buttons.append(
            '<button class="league-filter" type="button" '
            f'data-sport-filter="{html.escape(label, quote=True)}">{html.escape(label)}</button>'
        )
    return f'<div class="league-filters" aria-label="Sport filters">{"".join(buttons)}</div>'


def render_sports_sections(sports: dict) -> str:
    sections = sports.get("sports") if isinstance(sports.get("sports"), list) else []
    rendered = []
    for section in sections:
        label = str(section.get("label", "")).strip()
        games = section.get("games") if isinstance(section.get("games"), list) else []
        games_html = "".join(render_game_card(game) for game in games if isinstance(game, dict))
        if not games_html:
            games_html = f'<div class="league-empty">{html.escape(str(section.get("note", "No games today.")))}</div>'
        rendered.append(
            '<section class="league-section" '
            f'data-league="{html.escape(label, quote=True)}">'
            '<div class="league-head">'
            "<div>"
            f'<div class="kicker">{html.escape(str(section.get("full_name", label)))}</div>'
            f"<h3>{html.escape(label)}</h3>"
            "</div>"
            f'<span>{len(games)} games</span>'
            "</div>"
            f'<div class="games-list">{games_html}</div>'
            "</section>"
        )
    return "".join(rendered)


def render_sports_dashboard(sports: dict) -> str:
    generated = html.escape(str(sports.get("generated_at_ct", "")))
    return (
        '<div class="sports-dashboard">'
        f'<div class="sports-stats">{render_sports_summary(sports)}</div>'
        f'{render_sports_filters(sports)}'
        f'{render_sports_sections(sports)}'
        f'<p class="sports-footnote">Scores and spreads refresh during the daily build. Lines are informational and can move quickly. Last checked {generated}.</p>'
        "</div>"
    )


def render_sports_preview(sports: dict) -> str:
    games = all_sports_games(sports)[:3]
    if not games:
        summary = sports.get("summary") if isinstance(sports.get("summary"), dict) else {}
        return (
            '<p class="feature-copy">'
            f'{html.escape(str(summary.get("games", 0)))} games found today across World Cup, NFL, CFB, NCAAB, and NBA.'
            "</p>"
        )
    rows = []
    for game in games:
        rows.append(
            '<li>'
            f'<span>{html.escape(str(game.get("league", "")))}</span>'
            f'<strong>{html.escape(str(game.get("away_team", "")))} @ {html.escape(str(game.get("home_team", "")))}</strong>'
            f'<em>{html.escape(str(game.get("status_label", "")))}</em>'
            "</li>"
        )
    return f'<ul class="sports-preview">{"".join(rows)}</ul>'


def clean_html_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines()) + "\n"


def render_page(
    *,
    page_date: datetime,
    edition: int,
    markets: dict,
    trivia: dict,
    sports: dict,
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
    trivia_options = render_trivia_options(trivia)
    sports_dashboard = render_sports_dashboard(sports)
    sports_preview = render_sports_preview(sports)
    ticker_path_json = json.dumps(ticker_path)
    amd_chart_config = json.dumps(
        {
            "autosize": True,
            "symbol": "NASDAQ:AMD",
            "interval": "D",
            "timezone": "America/Chicago",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "backgroundColor": "#15191a",
            "gridColor": "rgba(255,255,255,0.06)",
            "allow_symbol_change": False,
            "calendar": False,
            "hide_side_toolbar": True,
            "hide_top_toolbar": False,
            "hide_legend": False,
            "save_image": False,
            "support_host": "https://www.tradingview.com",
        },
        indent=8,
    )

    return clean_html_output(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Daily Brief - {html.escape(page_date.date().isoformat())}</title>
  <link rel="preconnect" href="https://unpkg.com">
  <link rel="preconnect" href="https://s3.tradingview.com">
  <link rel="preconnect" href="https://www.tradingview.com">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0c0f0e;
      --panel: #15191a;
      --panel-2: #1c2224;
      --panel-3: #111516;
      --text: #f4f1ea;
      --muted: #98a39f;
      --line: #2b3433;
      --line-strong: #3d4946;
      --gold: #e7b95f;
      --green: #62d293;
      --teal: #66d9d0;
      --blue: #87a9ff;
      --red: #f07f7f;
      --shadow: 0 16px 42px rgba(0, 0, 0, 0.32);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      padding: 0 0 5rem;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      z-index: -1;
      background:
        linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.75), transparent 72%);
    }}
    a {{ color: var(--blue); }}
    .page {{
      width: min(100% - 32px, 1180px);
      margin: 0 auto;
      padding: 1rem 0 3.5rem;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 1rem;
      align-items: end;
      padding: 1.25rem 0 1rem;
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
      margin: 0.15rem 0 0;
      font-size: clamp(2.1rem, 7vw, 4.8rem);
      font-weight: 850;
      letter-spacing: 0;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 0.5rem;
    }}
    .status-pill {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
      padding: 0.55rem 0.7rem;
      min-height: 4rem;
    }}
    .status-label {{
      color: var(--muted);
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
    }}
    .status-value {{
      display: block;
      margin-top: 0.2rem;
      font-size: 0.96rem;
      font-weight: 750;
    }}
    .view-tabs {{
      display: inline-flex;
      gap: 0.25rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
      padding: 0.25rem;
      margin: 0 0 1rem;
    }}
    .tab-button {{
      min-height: 38px;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      padding: 0.45rem 0.8rem;
      font-size: 0.9rem;
    }}
    .tab-button[aria-selected="true"] {{
      background: var(--teal);
      color: #061514;
    }}
    .tab-panel[hidden] {{ display: none; }}
    .dashboard-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 1rem;
      min-width: 0;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}
    .panel-head h2 {{
      margin: 0;
      font-size: 1.05rem;
      font-weight: 800;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 1.55rem;
      border-radius: 8px;
      border: 1px solid rgba(231, 185, 95, 0.45);
      color: var(--gold);
      padding: 0.1rem 0.55rem;
      font-size: 0.76rem;
      white-space: nowrap;
    }}
    .markets-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 0.7rem;
    }}
    .markets-list li {{
      border-left: 2px solid rgba(102, 217, 208, 0.45);
      padding-left: 0.7rem;
      color: #e7ece8;
      font-size: 0.95rem;
    }}
    .stamp, .sources {{
      margin: 0.75rem 0 0;
      color: var(--muted);
      font-size: 0.8rem;
    }}
    .sources a {{ color: var(--muted); }}
    .amd-panel {{
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    .chart-frame {{
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #101313;
    }}
    .tradingview-widget-container,
    .tradingview-widget-container__widget {{
      width: 100%;
      height: 100%;
    }}
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
      background: var(--teal);
      color: #061514;
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
    .trivia-options {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.45rem;
      margin: 0 0 0.9rem;
      padding: 0;
      list-style: none;
    }}
    .trivia-options li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
      color: #dfe6e1;
      padding: 0.5rem 0.65rem;
      font-size: 0.92rem;
    }}
    .sports-preview {{
      display: grid;
      gap: 0.55rem;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .sports-preview li {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 0.6rem;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 0.4rem 0;
      min-width: 0;
    }}
    .sports-preview span {{
      color: var(--green);
      font-size: 0.78rem;
      font-weight: 800;
    }}
    .sports-preview strong {{
      overflow-wrap: anywhere;
      font-size: 0.92rem;
    }}
    .sports-preview em {{
      color: var(--muted);
      font-size: 0.82rem;
      font-style: normal;
      white-space: nowrap;
    }}
    .sports-layout {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }}
    .sports-dashboard {{
      display: grid;
      gap: 1rem;
    }}
    .sports-stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.55rem;
    }}
    .sports-stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
      padding: 0.65rem;
    }}
    .sports-stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }}
    .sports-stat strong {{
      display: block;
      margin-top: 0.15rem;
      font-size: 1.4rem;
      line-height: 1.1;
    }}
    .league-filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }}
    .league-filter {{
      min-height: 34px;
      border: 1px solid var(--line);
      background: var(--panel-3);
      color: var(--muted);
      padding: 0.35rem 0.65rem;
      font-size: 0.84rem;
    }}
    .league-filter.is-active {{
      background: var(--green);
      border-color: var(--green);
      color: #07150e;
    }}
    .league-section {{
      display: grid;
      gap: 0.7rem;
      border-top: 1px solid var(--line);
      padding-top: 0.9rem;
    }}
    .league-section.is-hidden {{ display: none; }}
    .league-head {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: end;
    }}
    .league-head h3 {{
      margin: 0.1rem 0 0;
      font-size: 1.3rem;
    }}
    .league-head > span {{
      color: var(--muted);
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .games-list {{
      display: grid;
      gap: 0.75rem;
    }}
    .game-card {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
      padding: 0.75rem;
      box-shadow: none;
    }}
    .game-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      align-items: center;
      justify-content: space-between;
    }}
    .league-tag,
    .game-status {{
      display: inline-flex;
      align-items: center;
      min-height: 1.45rem;
      border-radius: 8px;
      padding: 0.08rem 0.5rem;
      font-size: 0.72rem;
      font-weight: 850;
    }}
    .league-tag {{
      background: rgba(135, 169, 255, 0.15);
      color: var(--blue);
    }}
    .game-status {{
      border: 1px solid var(--line-strong);
      color: var(--muted);
    }}
    .status-live {{
      border-color: rgba(98, 210, 147, 0.45);
      color: var(--green);
    }}
    .status-final {{
      border-color: rgba(231, 185, 95, 0.45);
      color: var(--gold);
    }}
    .teams {{
      display: grid;
      gap: 0.3rem;
      min-width: 0;
    }}
    .team-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.75rem;
      align-items: baseline;
      min-height: 1.55rem;
    }}
    .team-row span {{
      min-width: 0;
      overflow-wrap: anywhere;
      font-weight: 750;
    }}
    .team-row strong {{
      min-width: 2ch;
      text-align: right;
      color: var(--gold);
      font-size: 1.08rem;
    }}
    .game-card.is-scoreless .team-row strong {{ display: none; }}
    .home-team span::before {{
      content: "@ ";
      color: var(--muted);
      font-weight: 700;
    }}
    .spread-box {{
      display: grid;
      gap: 0.35rem;
      border-left: 2px solid rgba(231, 185, 95, 0.55);
      padding-left: 0.65rem;
      color: #e7ece8;
    }}
    .spread-box > span {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }}
    .spread-box > strong {{
      color: var(--text);
      font-size: 0.94rem;
    }}
    .spread-line {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.65rem;
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .spread-line span:first-child {{
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .spread-price {{
      color: var(--muted);
      font-weight: 650;
    }}
    .spread-box small {{
      color: var(--muted);
      font-size: 0.75rem;
    }}
    .spread-empty {{
      border-left-color: var(--line-strong);
    }}
    .league-empty {{
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      color: var(--muted);
      padding: 0.85rem;
    }}
    .sports-footnote {{
      margin: 0;
      color: var(--muted);
      font-size: 0.8rem;
    }}
    .deep-link-card {{
      display: grid;
      gap: 0.75rem;
      align-content: start;
    }}
    .feature-title {{
      margin: 0.2rem 0 0.75rem;
      font-size: clamp(1.55rem, 5vw, 2.45rem);
      font-weight: 850;
    }}
    .feature-copy {{
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .text-button {{
      justify-self: start;
      display: inline-flex;
      align-items: center;
      min-height: 42px;
      border-radius: 8px;
      background: var(--teal);
      color: #061514;
      padding: 0.55rem 0.8rem;
      text-decoration: none;
      font-weight: 800;
    }}
    .map-wrap {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }}
    #history-map {{
      display: none;
      width: 100%;
      height: min(55vh, 410px);
      min-height: 280px;
    }}
    .map-wrap.is-ready #history-map {{ display: block; }}
    .map-wrap.is-ready .map-fallback {{ display: none; }}
    .map-fallback {{
      padding: 0.95rem;
    }}
    .map-fallback-title {{
      margin-bottom: 0.65rem;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
    }}
    .map-fallback ul {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 0.7rem;
    }}
    .map-fallback li {{
      display: grid;
      grid-template-columns: 0.8rem 1fr;
      gap: 0.55rem;
      align-items: start;
    }}
    .map-dot {{
      width: 0.72rem;
      height: 0.72rem;
      margin-top: 0.47rem;
      border-radius: 999px;
      background: var(--green);
    }}
    .map-dot-capital {{ background: var(--gold); }}
    .map-dot-battle {{ background: var(--red); }}
    .map-dot-city {{ background: var(--blue); }}
    .map-caption {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
      margin-top: 0.08rem;
    }}
    .deep-read-layout {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }}
    .read-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
      justify-content: space-between;
      align-items: flex-end;
      margin-bottom: 1rem;
    }}
    .read-title {{
      margin: 0.2rem 0 0;
      font-size: clamp(1.75rem, 5vw, 3.4rem);
      font-weight: 850;
    }}
    .read-map-panel {{
      display: grid;
      gap: 1rem;
      align-content: start;
    }}
    article {{
      width: min(100%, 820px);
      color: #e8e4dc;
      font-size: 1.05rem;
      line-height: 1.72;
    }}
    article h2 {{
      font-size: 1.25rem;
      margin: 1.7rem 0 0.45rem;
    }}
    article h3 {{
      font-size: 1.08rem;
      color: var(--teal);
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
      background: #070908;
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
      background: var(--teal);
      color: #061514;
      font-weight: 850;
      font-size: 0.78rem;
    }}
    .ticker-window {{ overflow: hidden; flex: 1; }}
    .ticker-track {{
      display: inline-flex;
      gap: 2rem;
      white-space: nowrap;
      animation: marquee 120s linear infinite;
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
      .panel {{ padding: 0.9rem; }}
      .panel-head {{ align-items: flex-start; }}
      .sports-preview li {{
        grid-template-columns: 1fr;
        gap: 0.15rem;
      }}
      .sports-preview em {{ white-space: normal; }}
      article {{ font-size: 1.06rem; }}
      .ticker-label {{ padding: 0 0.55rem; }}
      .ticker-track {{ gap: 1.4rem; animation-duration: 115s; }}
    }}
    @media (min-width: 780px) {{
      .topbar {{
        grid-template-columns: minmax(0, 1fr) minmax(320px, 430px);
      }}
      .status-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
      .dashboard-grid {{
        grid-template-columns: repeat(12, minmax(0, 1fr));
        align-items: stretch;
      }}
      .markets-panel {{ grid-column: span 7; }}
      .amd-panel {{ grid-column: span 5; }}
      .sports-peek-panel {{ grid-column: span 4; }}
      .deep-link-card {{ grid-column: span 8; }}
      .chart-frame {{ height: 430px; }}
      .sports-layout {{
        grid-template-columns: repeat(12, minmax(0, 1fr));
        align-items: start;
      }}
      .sports-main-panel {{ grid-column: span 8; }}
      .trivia-panel {{ grid-column: span 4; }}
      .sports-stats {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      .game-card {{
        grid-template-columns: minmax(0, 1fr) minmax(220px, 0.55fr);
        align-items: start;
      }}
      .game-meta {{
        grid-column: 1 / -1;
      }}
      .deep-read-layout {{
        grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
        align-items: start;
      }}
      .read-map-panel {{
        position: sticky;
        top: 1rem;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div>
        <div class="kicker">Personal command center</div>
        <h1>Daily Brief</h1>
      </div>
      <div class="status-grid" aria-label="Edition status">
        <div class="status-pill">
          <span class="status-label">Today</span>
          <span class="status-value">{html.escape(human_date(page_date))}</span>
        </div>
        <div class="status-pill">
          <span class="status-label">Edition</span>
          <span class="status-value">#{edition}</span>
        </div>
        <div class="status-pill">
          <span class="status-label">Market Card</span>
          <span class="status-value">{html.escape(str(markets.get("generated_at_ct", "")))}</span>
        </div>
      </div>
    </header>

    <nav class="view-tabs" aria-label="Daily views">
      <button class="tab-button" type="button" id="overview-tab" aria-selected="true" aria-controls="overview-panel" data-tab-target="overview-panel">Overview</button>
      <button class="tab-button" type="button" id="sports-tab" aria-selected="false" aria-controls="sports-panel" data-tab-target="sports-panel">Sports</button>
      <button class="tab-button" type="button" id="deep-read-tab" aria-selected="false" aria-controls="deep-read-panel" data-tab-target="deep-read-panel">Deep Read</button>
    </nav>

    <section class="tab-panel" id="overview-panel" role="tabpanel" aria-labelledby="overview-tab">
      <section class="dashboard-grid" aria-label="Daily dashboard">
        <section class="panel markets-panel" aria-labelledby="markets-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Markets</div>
              <h2 id="markets-heading">US Market Signals</h2>
            </div>
            {render_badge(markets.get("badge"))}
          </div>
          <ul class="markets-list">
            {markets_items}
          </ul>
          {sources_html}
        </section>

        <section class="panel amd-panel" aria-labelledby="amd-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Watchlist</div>
              <h2 id="amd-heading">AMD Live Chart</h2>
            </div>
            <span class="badge">NASDAQ: AMD</span>
          </div>
          <div class="chart-frame">
            <div class="tradingview-widget-container" aria-label="AMD live stock chart">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                {amd_chart_config}
              </script>
            </div>
          </div>
        </section>

        <section class="panel sports-peek-panel" aria-labelledby="sports-peek-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Sports</div>
              <h2 id="sports-peek-heading">Today's Slate</h2>
            </div>
            {render_badge(sports.get("badge"))}
          </div>
          {sports_preview}
          <button class="text-button" type="button" data-tab-target="sports-panel">Open Sports</button>
        </section>

        <section class="panel deep-link-card" aria-labelledby="deep-link-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Historical Read</div>
              <h2 id="deep-link-heading">Today's Deep Read</h2>
            </div>
            {render_badge(article.get("badge"))}
          </div>
          <h3 class="feature-title">{html.escape(str(article.get("title", "")))}</h3>
          <p class="feature-copy">Long-form history, spatial context, and the full daily narrative.</p>
          <button class="text-button" type="button" data-tab-target="deep-read-panel">Open Deep Read</button>
        </section>
      </section>
    </section>

    <section class="tab-panel" id="sports-panel" role="tabpanel" aria-labelledby="sports-tab" hidden>
      <section class="sports-layout" aria-label="Sports dashboard">
        <section class="panel sports-main-panel" aria-labelledby="sports-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Scores & Spreads</div>
              <h2 id="sports-heading">Today's Sports Slate</h2>
            </div>
            {render_badge(sports.get("badge"))}
          </div>
          {sports_dashboard}
        </section>

        <section class="panel trivia-panel" aria-labelledby="trivia-heading">
          <div class="panel-head">
            <div>
              <div class="kicker">Trivia</div>
              <h2 id="trivia-heading">Daily Sports Question</h2>
            </div>
            {render_badge(trivia.get("badge"))}
          </div>
          <div class="trivia-meta">{html.escape(str(trivia.get("sport", "")))} · {html.escape(str(trivia.get("difficulty", "")).title())}</div>
          <p class="question">{trivia_question}</p>
          {trivia_options}
          <button id="reveal-answer" type="button" aria-expanded="false" aria-controls="trivia-answer">Reveal Answer</button>
          <div id="trivia-answer" class="answer">
            <p><strong>{trivia_answer}</strong></p>
            <p>{trivia_story}</p>
          </div>
        </section>
      </section>
    </section>

    <section class="tab-panel" id="deep-read-panel" role="tabpanel" aria-labelledby="deep-read-tab" hidden>
      <div class="deep-read-layout">
        <aside class="panel read-map-panel" aria-labelledby="article-heading">
          <div class="read-header">
            <div>
              <div class="kicker">Deep Read</div>
              <h2 id="article-heading" class="read-title">{html.escape(str(article.get("title", "")))}</h2>
            </div>
            <a href="{archive_prefix}">Archive</a>
          </div>
          {map_html}
        </aside>
        <article class="panel">
          {article_html}
        </article>
      </div>
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

    const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
    const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
    const setActiveTab = (targetId) => {{
      tabPanels.forEach((panel) => {{
        panel.hidden = panel.id !== targetId;
      }});
      tabButtons.forEach((button) => {{
        const selected = button.dataset.tabTarget === targetId;
        if (button.classList.contains('tab-button')) {{
          button.setAttribute('aria-selected', String(selected));
        }}
      }});
      if (targetId === 'deep-read-panel') {{
        initializeHistoryMap();
      }}
    }};
    tabButtons.forEach((button) => {{
      button.addEventListener('click', () => setActiveTab(button.dataset.tabTarget));
    }});

    const answerButton = document.getElementById('reveal-answer');
    const answer = document.getElementById('trivia-answer');
    answerButton?.addEventListener('click', () => {{
      const visible = answer.classList.toggle('is-visible');
      answerButton.setAttribute('aria-expanded', String(visible));
      answerButton.textContent = visible ? 'Hide Answer' : 'Reveal Answer';
    }});

    const sportFilters = Array.from(document.querySelectorAll('[data-sport-filter]'));
    const leagueSections = Array.from(document.querySelectorAll('[data-league]'));
    sportFilters.forEach((filter) => {{
      filter.addEventListener('click', () => {{
        const target = filter.dataset.sportFilter;
        sportFilters.forEach((item) => item.classList.toggle('is-active', item === filter));
        leagueSections.forEach((section) => {{
          section.classList.toggle('is-hidden', target !== 'all' && section.dataset.league !== target);
        }});
      }});
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
    const mapWrap = mapEl?.closest('[data-map-wrap]');
    let historyMap = null;
    const showMapFallback = () => {{
      mapWrap?.classList.remove('is-ready');
      mapWrap?.classList.add('is-fallback');
    }};
    function initializeHistoryMap() {{
      if (historyMap) {{
        setTimeout(() => historyMap.invalidateSize(), 0);
        return;
      }}
      if (!mapEl) {{
        return;
      }}
      try {{
        if (!window.L) {{
          throw new Error('Leaflet did not load');
        }}
        const locations = JSON.parse(mapEl.dataset.locations || '[]');
        if (!locations.length) {{
          throw new Error('No map locations');
        }}
        mapWrap?.classList.add('is-ready');
        historyMap = L.map(mapEl, {{ scrollWheelZoom: false }});
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
          maxZoom: 19,
          subdomains: 'abcd',
          attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }}).addTo(historyMap);
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
          }}).addTo(historyMap);
          marker.bindPopup(`<strong>${{escapeHtml(loc.name)}}</strong><br>${{escapeHtml(loc.caption || '')}}`);
          bounds.push([loc.lat, loc.lng]);
        }});
        historyMap.fitBounds(bounds, {{ padding: [28, 28], maxZoom: 6 }});
        setTimeout(() => historyMap.invalidateSize(), 0);
      }} catch (error) {{
        showMapFallback();
      }}
    }}
  </script>
</body>
</html>
""")


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
    if not isinstance(trivia_history, list):
        trivia_history = []

    if not topics:
        raise RuntimeError("topics.json has no topics.")

    edition = int(state.get("edition", 0)) + 1
    topic, next_topic = choose_topic(topics)

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
        trivia = generate_trivia(trivia_history)
        print("Fetched sports trivia.")
    except Exception as exc:
        print(f"Trivia fetch failed: {exc}")
        trivia = demo_trivia()
    trivia_history = remember_trivia(trivia_history, today_key, trivia)

    try:
        sports = generate_sports_dashboard(now)
        print("Fetched sports scores and spreads.")
    except Exception as exc:
        print(f"Sports feed failed: {exc}")
        sports = reuse_previous(state, "sports", "previous slate", str(exc)) or demo_sports(now, str(exc))

    try:
        update_ticker()
    except Exception as exc:
        print(f"Ticker update failed: {exc}. Continuing with previous ticker if available.")

    page = render_page(
        page_date=now,
        edition=edition,
        markets=markets,
        trivia=trivia,
        sports=sports,
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
            sports=sports,
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
    update_last_successful(state, "sports", sports)

    if len(unused_topics(topics)) < 10:
        try:
            if append_fresh_topics(topics):
                print("Topic queue refreshed.")
        except Exception as exc:
            print(f"Topic refresh failed: {exc}")

    write_json(STATE_FILE, state)
    write_json(TOPICS_FILE, topics)
    write_json(TRIVIA_FILE, trivia_history)
    write_json(SITE_DIR / "sports.json", sports)
    print(f"Wrote {index_path.relative_to(ROOT)} and archive page {archive_path.relative_to(ROOT)}.")


def render_current_site() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    now = today_central()
    today_key = now.date().isoformat()
    state = load_json(STATE_FILE, {"edition": 0, "last_successful": {}})
    topics = load_json(TOPICS_FILE, [])
    trivia_history = load_json(TRIVIA_FILE, [])
    if not isinstance(trivia_history, list):
        trivia_history = []

    fallback_topic = topics[0] if topics else {"title": "History", "era": "", "region": ""}
    fallback_next_topic = topics[1] if len(topics) > 1 else fallback_topic
    article = copy.deepcopy(state.get("last_successful", {}).get("article")) or demo_article(
        fallback_topic,
        fallback_next_topic,
    )
    markets = copy.deepcopy(state.get("last_successful", {}).get("markets")) or demo_markets(now)

    try:
        trivia = generate_trivia(trivia_history)
    except Exception as exc:
        print(f"Trivia fetch failed while rendering current site: {exc}")
        trivia = demo_trivia()

    try:
        sports = generate_sports_dashboard(now)
    except Exception as exc:
        print(f"Sports feed failed while rendering current site: {exc}")
        sports = demo_sports(now, str(exc))

    edition = int(state.get("edition", 0))
    index_path = SITE_DIR / "index.html"
    archive_path = ARCHIVE_DIR / f"{today_key}.html"
    index_path.write_text(
        render_page(
            page_date=now,
            edition=edition,
            markets=markets,
            trivia=trivia,
            sports=sports,
            article=article,
            ticker_path="ticker.json",
        ),
        encoding="utf-8",
    )
    archive_path.write_text(
        render_page(
            page_date=now,
            edition=edition,
            markets=markets,
            trivia=trivia,
            sports=sports,
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
    write_json(SITE_DIR / "sports.json", sports)
    print(f"Re-rendered {index_path.relative_to(ROOT)} and archive page {archive_path.relative_to(ROOT)}.")


if __name__ == "__main__":
    if "--render-current" in sys.argv:
        render_current_site()
    else:
        main()
