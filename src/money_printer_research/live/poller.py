"""One poll: today's games from ESPN into the live database, re-predicted when the XIs land.

For each league's games today:

1. match the ESPN teams to an Understat fixture (league, day, both teams)
2. upsert the game: state, ESPN's status line and score
3. once per game, store the "fixture" prediction: predict's features, guessed XI
4. from lineup_window_minutes before kick-off until the end, ask ESPN for the XIs once
   they are not stored yet; when both are there, store them with each starter's
   Understat player and form, and the "lineup" prediction from the real XIs

The model and the tables it reads are loaded once and reloaded when `modeling` or
`clean_data` rewrites them. Without them (no model yet) games and line-ups are still
stored, without predictions.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from money_printer_research.config import Settings
from money_printer_research.live import espn
from money_printer_research.live.lineup import (
    current_form,
    match_starters,
    match_team,
    with_lineup,
)
from money_printer_research.live.store import Store

log = logging.getLogger(__name__)


@dataclass
class Predictor:
    """The saved model with what it needs, loaded lazily and reloaded when files change."""

    settings: Settings
    _stamp: tuple[float, ...] | None = None
    _failed: tuple[float, ...] | None = None
    _parts: dict[str, Any] = field(default_factory=dict)

    def _files(self) -> list[Path]:
        out = self.settings.clean.out_dir
        return [
            self.settings.modeling.out_file,
            out / "match_features.csv",
            out / "player_stat.csv",
            out / "cleaned_matches.csv",
            self.settings.understat.schedule_file,
        ]

    def load(self) -> dict[str, Any] | None:
        """model, history, fixtures and form; None when a file is missing or fails to load.

        A failed load is logged once and retried when one of the files changes, so the
        games are still stored, without predictions, in the meantime.
        """
        files = self._files()
        if not all(f.is_file() for f in files):
            return None
        stamp = tuple(f.stat().st_mtime for f in files)
        if stamp == self._failed:
            return None
        if stamp != self._stamp:
            try:
                self._parts = self._read(files)
            except Exception:
                log.exception("could not load the model and its tables; no predictions")
                self._failed = stamp
                return None
            self._stamp = stamp
        return self._parts

    def _read(self, files: list[Path]) -> dict[str, Any]:
        from money_printer_research.analysis.modeling import load_model
        from money_printer_research.analysis.predict import upcoming
        from money_printer_research.clean_data.rename_team_names import norm_team_names
        from money_printer_research.schema import read_clean

        out = self.settings.clean.out_dir
        player_stat = norm_team_names(read_clean(out, "player_stat"))
        return {
            "model": load_model(files[0]),
            "history": read_clean(out, "match_features"),
            "fixtures": upcoming(self.settings, datetime.now(UTC).date() - timedelta(days=1)),
            "form": current_form(player_stat, read_clean(out, "cleaned_matches")),
        }

    def fixtures(self) -> pl.DataFrame | None:
        """Unplayed Understat fixtures from yesterday on, or None without the data."""
        parts = self.load()
        return parts["fixtures"] if parts else None

    def features(self, game_id: int) -> pl.DataFrame | None:
        """The feature row predict builds for one fixture (guessed XI), or None."""
        from money_printer_research.analysis.predict import fixture_features

        parts = self.load()
        if parts is None:
            return None
        fixture = parts["fixtures"].filter(pl.col("game_id") == game_id)
        return fixture_features(parts["history"], fixture) if fixture.height else None

    def predict(self, frame: pl.DataFrame) -> dict[str, float]:
        """p_H, p_D and p_A of the first row of `frame`."""
        row = self._parts["model"].predict(frame).row(0, named=True)
        return {k: float(row[k]) for k in ("p_H", "p_D", "p_A")}

    def form(self) -> pl.DataFrame | None:
        """Every player's current_form, or None without the data."""
        parts = self.load()
        return parts["form"] if parts else None


def find_fixture(game: espn.Game, fixtures: pl.DataFrame | None) -> dict[str, Any]:
    """Understat home_team, away_team and game_id for an ESPN game; None where unmatched."""
    none = {"home_team": None, "away_team": None, "game_id": None}
    if fixtures is None:
        return none
    day = fixtures.filter(
        (pl.col("league") == game.league) & (pl.col("date").dt.date() == game.kickoff.date())
    )
    teams = sorted(set(day["home_team"]) | set(day["away_team"]))
    home, away = match_team(game.home_team, teams), match_team(game.away_team, teams)
    hit = day.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
    return {
        "home_team": home,
        "away_team": away,
        "game_id": int(hit["game_id"][0]) if hit.height else None,
    }


def wants_lineup(game: espn.Game, now: datetime, window: timedelta) -> bool:
    """From `window` before kick-off until the final whistle."""
    return game.state == "in" or (game.state == "pre" and game.kickoff - now <= window)


@dataclass
class Fetchers:
    """ESPN calls, swapped out in tests."""

    games_on: Callable[[str, date], list[espn.Game]] = espn.games_on
    lineups: Callable[[str, str], list[espn.Starter]] = espn.lineups


def _poll_game(
    game: espn.Game,
    store: Store,
    predictor: Predictor,
    fetch: Fetchers,
    now: datetime,
    window: timedelta,
) -> None:
    ids = find_fixture(game, predictor.fixtures())
    store.upsert_game(
        {
            "espn_id": game.espn_id,
            "league": game.league,
            "kickoff": game.kickoff.isoformat(),
            "state": game.state,
            "detail": game.detail,
            "espn_home": game.home_team,
            "espn_away": game.away_team,
            "home_score": game.home_score,
            "away_score": game.away_score,
            **ids,
        }
    )
    need_fixture = not store.has_prediction(game.espn_id, "fixture")
    need_lineup = not store.has_lineup(game.espn_id) and wants_lineup(game, now, window)
    if not (need_fixture or need_lineup):
        return
    features = predictor.features(ids["game_id"]) if ids["game_id"] is not None else None
    if features is not None and need_fixture:
        store.set_prediction(game.espn_id, "fixture", predictor.predict(features))
    if not need_lineup:
        return
    starters = fetch.lineups(game.league, game.espn_id)
    if not espn.has_both_xis(starters):
        return
    rows = [{"side": s.side, "name": s.name, "position": s.position} for s in starters]
    form = predictor.form()
    if form is not None and ids["home_team"] and ids["away_team"]:
        rows = match_starters(rows, form, ids["home_team"], ids["away_team"])
    store.set_lineup(game.espn_id, rows)
    log.info("line-ups in for %s v %s", game.home_team, game.away_team)
    if features is not None and form is not None:
        store.set_prediction(
            game.espn_id, "lineup", predictor.predict(with_lineup(features, rows))
        )


def refresh(
    settings: Settings,
    store: Store,
    predictor: Predictor,
    fetch: Fetchers | None = None,
    now: datetime | None = None,
) -> int:
    """Poll every league once; return how many games were seen. A failing league is logged."""
    fetch = fetch or Fetchers()
    now = now or datetime.now(UTC)
    window = timedelta(minutes=settings.live.lineup_window_minutes)
    seen = 0
    for league in espn.SLUGS:
        try:
            for game in fetch.games_on(league, now.date()):
                _poll_game(game, store, predictor, fetch, now, window)
                seen += 1
        except Exception:
            log.exception("live poll failed for %s", league)
    return seen
