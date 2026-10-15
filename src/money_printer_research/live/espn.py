"""ESPN's public site API: today's games with their state and score, and the starting XIs.

    scoreboard  .../soccer/<slug>/scoreboard?dates=YYYYMMDD   games, state, clock, score
    summary     .../soccer/<slug>/summary?event=<id>          rosters, starter flag per player

ESPN posts the XIs when the clubs announce them, about an hour before kick-off. The
parse_* functions only read the JSON, so they are tested on saved responses.
"""

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

SLUGS = {
    "ENG-Premier League": "eng.1",
    "ESP-La Liga": "esp.1",
    "GER-Bundesliga": "ger.1",
    "ITA-Serie A": "ita.1",
    "FRA-Ligue 1": "fra.1",
}
"""Understat league name to ESPN league slug."""

TIMEOUT = 20


@dataclass(frozen=True)
class Game:
    """One game on ESPN's scoreboard."""

    espn_id: str
    league: str
    kickoff: datetime
    state: str
    """"pre" (not started), "in" (playing) or "post" (finished)."""
    detail: str
    """ESPN's short status: "FT", "HT", "67'", or the kick-off time before it starts."""
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None


@dataclass(frozen=True)
class Starter:
    """One player in a starting XI."""

    side: str
    """"home" or "away"."""
    name: str
    position: str


def fetch_json(url: str) -> dict:
    """GET `url` and parse the JSON body."""
    req = urllib.request.Request(url, headers={"User-Agent": "money-printer-research"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def scoreboard_url(league: str, day: date) -> str:
    return f"{BASE}/{SLUGS[league]}/scoreboard?dates={day:%Y%m%d}"


def summary_url(league: str, espn_id: str) -> str:
    return f"{BASE}/{SLUGS[league]}/summary?event={espn_id}"


def _score(competitor: dict, state: str) -> int | None:
    """The side's goals, or None before kick-off (ESPN sends "0" then)."""
    if state == "pre" or competitor.get("score") in (None, ""):
        return None
    return int(competitor["score"])


def parse_scoreboard(body: dict, league: str) -> list[Game]:
    """Every game in a scoreboard response."""
    games = []
    for event in body.get("events", []):
        comp = event["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        status = event["status"]["type"]
        state = status["state"]
        games.append(
            Game(
                espn_id=str(event["id"]),
                league=league,
                kickoff=datetime.fromisoformat(event["date"].replace("Z", "+00:00")),
                state=state,
                detail=status.get("shortDetail", ""),
                home_team=sides["home"]["team"]["displayName"],
                away_team=sides["away"]["team"]["displayName"],
                home_score=_score(sides["home"], state),
                away_score=_score(sides["away"], state),
            )
        )
    return games


def parse_lineups(body: dict) -> list[Starter]:
    """Both starting XIs from a summary response; empty until ESPN has them."""
    starters = []
    for roster in body.get("rosters", []):
        for player in roster.get("roster", []):
            if player.get("starter"):
                starters.append(
                    Starter(
                        side=roster["homeAway"],
                        name=player["athlete"]["displayName"],
                        position=player.get("position", {}).get("abbreviation", ""),
                    )
                )
    return starters


def has_both_xis(starters: list[Starter]) -> bool:
    """True when ESPN lists eleven starters for each side."""
    return all(sum(s.side == side for s in starters) == 11 for side in ("home", "away"))


def games_on(league: str, day: date) -> list[Game]:
    return parse_scoreboard(fetch_json(scoreboard_url(league, day)), league)


def lineups(league: str, espn_id: str) -> list[Starter]:
    return parse_lineups(fetch_json(summary_url(league, espn_id)))
