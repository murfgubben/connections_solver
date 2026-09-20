"""Fetch and cache NYT Connections puzzles."""

from __future__ import annotations

import json
import logging
import time
from datetime import date as date_type
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://www.nytimes.com/games/connections"
API_URL = "https://www.nytimes.com/svc/connections/v2/{date}.json"
DATA_DIR = Path(__file__).resolve().parent / "data"
USER_AGENT = "connections-solver/0.1 (local research project)"
REQUEST_DELAY_SECONDS = 0.5


def _normalise_date(value: str | None) -> str:
    if value is None:
        return date_type.today().isoformat()
    try:
        return date_type.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"date must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def save_puzzle(puzzle: dict[str, Any], directory: Path = DATA_DIR) -> Path:
    """Save a validated puzzle as ``data/{date}.json`` and return its path."""
    puzzle_date = puzzle.get("date")
    if not isinstance(puzzle_date, str):
        raise ValueError("cannot cache puzzle without a string date")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{puzzle_date}.json"
    path.write_text(json.dumps(puzzle, indent=2) + "\n", encoding="utf-8")
    return path


def _load_cached(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        puzzle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read cached puzzle {path}: {exc}") from exc
    _validate_puzzle(puzzle)
    return puzzle


def _validate_puzzle(puzzle: Any) -> None:
    if not isinstance(puzzle, dict) or not isinstance(puzzle.get("date"), str):
        raise RuntimeError("NYT response is missing a valid puzzle date")
    words = puzzle.get("words")
    answers = puzzle.get("answer")
    if not isinstance(words, list) or len(words) != 16 or not all(isinstance(w, str) for w in words):
        raise RuntimeError("NYT response did not contain exactly 16 word strings")
    if not isinstance(answers, list) or len(answers) != 4:
        raise RuntimeError("NYT response did not contain exactly four answer groups")
    for group in answers:
        if not isinstance(group, dict) or not isinstance(group.get("category"), str):
            raise RuntimeError("NYT answer group is missing its category")
        if len(group.get("words", [])) != 4:
            raise RuntimeError("NYT answer group did not contain exactly four words")


def _parse_api_response(payload: Any, requested_date: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("print_date") != requested_date:
        raise RuntimeError("NYT API response has an unexpected date or shape")
    categories = payload.get("categories")
    if not isinstance(categories, list) or len(categories) != 4:
        raise RuntimeError("NYT API response did not contain four categories")

    difficulties = ("yellow", "green", "blue", "purple")
    answer = []
    words = []
    for index, category in enumerate(categories):
        cards = category.get("cards") if isinstance(category, dict) else None
        title = category.get("title") if isinstance(category, dict) else None
        if not isinstance(title, str) or not isinstance(cards, list) or len(cards) != 4:
            raise RuntimeError("NYT API category is missing a title or four cards")
        group_words = [card.get("content") if isinstance(card, dict) else None for card in cards]
        if not all(isinstance(word, str) and word for word in group_words):
            raise RuntimeError("NYT API category contains an invalid card")
        words.extend(group_words)
        answer.append({"category": title, "difficulty": difficulties[index], "words": group_words})

    puzzle = {"date": requested_date, "words": words, "answer": answer}
    _validate_puzzle(puzzle)
    return puzzle


def scrape_connections(date: str | None = None) -> dict[str, Any]:
    """Load a cached puzzle or fetch and cache one from the NYT Connections API."""
    puzzle_date = _normalise_date(date)
    cached = _load_cached(DATA_DIR / f"{puzzle_date}.json")
    if cached is not None:
        return cached

    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        page = requests.get(BASE_URL, headers=headers, timeout=20)
        page.raise_for_status()
        # Keep the static-page check explicit: the current page sets gameData to undefined
        # and loads the actual puzzle through the API below.
        soup = BeautifulSoup(page.text, "html.parser")
        game_data = next(
            (script.get_text() for script in soup.find_all("script")
             if "window.gameData" in script.get_text()),
            None,
        )
        if game_data and "undefined" not in game_data:
            LOGGER.info("NYT page contains embedded game data; API remains the stable parser")

        response = requests.get(API_URL.format(date=puzzle_date), headers=headers, timeout=20)
        response.raise_for_status()
        puzzle = _parse_api_response(response.json(), puzzle_date)
    except requests.RequestException as exc:
        raise RuntimeError(f"could not fetch NYT Connections puzzle for {puzzle_date}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"NYT returned invalid JSON for {puzzle_date}") from exc

    save_puzzle(puzzle)
    return puzzle
