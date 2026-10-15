from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from money_printer_research.config import LiveSettings, Settings
from money_printer_research.live import espn
from money_printer_research.live.lineup import (
    current_form,
    match_player,
    match_starters,
    match_team,
    plain,
    with_lineup,
)
from money_printer_research.live.poller import Fetchers, Predictor, find_fixture, refresh
from money_printer_research.live.store import Store
from money_printer_research.schema import slot_columns

KICKOFF = datetime(2026, 10, 17, 14, 0, tzinfo=UTC)


def _event(espn_id: str, state: str, home: str, away: str, score=("0", "0")) -> dict:
    return {
        "id": espn_id,
        "date": "2026-10-17T14:00Z",
        "status": {
            "type": {
                "state": state,
                "shortDetail": {"pre": "2:00 PM", "in": "67'"}.get(state, "FT"),
            }
        },
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": home}, "score": score[0]},
                    {"homeAway": "away", "team": {"displayName": away}, "score": score[1]},
                ]
            }
        ],
    }


def _summary(home: list[str], away: list[str]) -> dict:
    def roster(side: str, names: list[str]) -> dict:
        players = [
            {"starter": i < 11, "athlete": {"displayName": n}, "position": {"abbreviation": "M"}}
            for i, n in enumerate(names)
        ]
        return {"homeAway": side, "roster": players}

    return {"rosters": [roster("home", home), roster("away", away)]}


# --- espn -------------------------------------------------------------------------


def test_parse_scoreboard_reads_state_and_score() -> None:
    body = {
        "events": [
            _event("1", "pre", "Everton", "Chelsea"),
            _event("2", "in", "Fulham", "Hull City", ("2", "1")),
        ]
    }
    pre, live = espn.parse_scoreboard(body, "ENG-Premier League")
    assert (pre.state, pre.home_score, pre.away_score) == ("pre", None, None)
    assert (live.state, live.detail, live.home_score, live.away_score) == ("in", "67'", 2, 1)
    assert live.kickoff == KICKOFF


def test_parse_lineups_keeps_only_starters() -> None:
    body = _summary([f"H{i}" for i in range(18)], [f"A{i}" for i in range(18)])
    starters = espn.parse_lineups(body)
    assert len(starters) == 22
    assert espn.has_both_xis(starters)
    assert not espn.has_both_xis(starters[:21])
    assert espn.parse_lineups({}) == []


# --- lineup -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("espn_name", "understat"),
    [
        ("Brighton & Hove Albion", "brighton"),
        ("AFC Bournemouth", "bournemouth"),
        ("Tottenham Hotspur", "tottenham"),
        ("Wolverhampton Wanderers", "wolverhampton wanderers"),
        ("Hull City", "hull city"),
    ],
)
def test_match_team_finds_the_understat_name(espn_name: str, understat: str) -> None:
    candidates = [
        "brighton",
        "bournemouth",
        "tottenham",
        "wolverhampton wanderers",
        "hull city",
        "chelsea",
    ]
    assert match_team(espn_name, candidates) == understat


def test_match_team_is_none_when_nothing_is_close() -> None:
    assert match_team("Real Madrid", ["chelsea", "everton"]) is None


def _player_stat() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two players over 12 games: Sánchez always 90 minutes, Kane 30."""
    games = list(range(1, 13))
    matches = pl.DataFrame(
        {"game_id": games, "date": [datetime(2026, 8, 1) + timedelta(days=7 * g) for g in games]}
    )
    rows = []
    for g in games:
        rows.append(("Alexis Sánchez", 7, "chelsea", g, 90, 0.5 if g > 2 else 5.0, 0.1))
        rows.append(("Harry Kane", 9, "chelsea", g, 30, 0.2, 0.0))
    stat = pl.DataFrame(
        rows,
        schema=["player", "player_id", "team", "game_id", "minutes", "xg", "xa"],
        orient="row",
    )
    return stat, matches


def test_current_form_sums_the_last_ten_games_including_the_last() -> None:
    form = current_form(*_player_stat()).sort("player_id")
    sanchez, kane = form.to_dicts()
    # Games 3-12: ten games of 90 minutes, 0.5 xG and 0.1 xA each.
    assert sanchez["prev_min"] == 900
    assert sanchez["form90"] == pytest.approx(0.6)
    # 300 minutes clears MIN_PREV_MIN (270).
    assert kane["prev_min"] == 300
    assert kane["xg90"] == pytest.approx(2.0 * 90 / 300)


def test_current_form_rates_are_null_below_the_minutes_floor() -> None:
    stat, matches = _player_stat()
    form = current_form(stat.filter(pl.col("game_id") > 10), matches)
    assert form["form90"].to_list() == [None, None]  # 180 and 60 minutes, both under 270


def test_match_player_handles_accents_and_initials() -> None:
    form = current_form(*_player_stat())
    assert match_player("Alexis Sanchez", form)["player_id"] == 7  # pyright: ignore[reportOptionalSubscript]
    assert match_player("H. Kane", form)["player_id"] == 9  # pyright: ignore[reportOptionalSubscript]
    assert match_player("Someone Else", form) is None
    assert plain("Kylian Mbappé-Lottin") == "kylian mbappe lottin"


def test_match_starters_and_with_lineup_fill_the_slots() -> None:
    form = current_form(*_player_stat())
    starters = [{"side": "home", "name": "Alexis Sánchez", "position": "F"}] + [
        {"side": "away", "name": "Nobody", "position": "G"}
    ]
    matched = match_starters(starters, form, "chelsea", "everton")
    assert matched[0]["player_id"] == 7 and matched[0]["prev_min"] == 900
    assert matched[1]["player_id"] is None and matched[1]["form90"] is None

    features = pl.DataFrame({"game_id": [1]}).with_columns(
        pl.lit(9.0).alias(c) for c in slot_columns()
    )
    out = with_lineup(features, matched).row(0, named=True)
    assert out["h_player1_prev_min"] == 900
    assert out["h_player2_prev_min"] is None  # only one home starter given
    assert out["a_player1_form90"] is None
    assert out["h_player1_fatigue_score"] is None


# --- store ------------------------------------------------------------------------


def _game(**kw: Any) -> dict[str, Any]:
    return {
        "espn_id": "1",
        "league": "ENG-Premier League",
        "kickoff": KICKOFF.isoformat(),
        "state": "pre",
        "detail": "2:00 PM",
        "espn_home": "Everton",
        "espn_away": "Chelsea",
        "home_team": "everton",
        "away_team": "chelsea",
        "game_id": 99,
        "home_score": None,
        "away_score": None,
        **kw,
    }


def test_store_upsert_keeps_the_matched_fixture(tmp_path: Path) -> None:
    store = Store(tmp_path / "live.sqlite")
    store.upsert_game(_game())
    store.upsert_game(
        _game(state="post", home_score=1, away_score=0, game_id=None, home_team=None)
    )
    (g,) = store.games()
    assert (g["state"], g["home_score"], g["game_id"], g["home_team"]) == (
        "post",
        1,
        99,
        "everton",
    )


def test_store_joins_both_predictions_and_the_lineup(tmp_path: Path) -> None:
    store = Store(tmp_path / "live.sqlite")
    store.upsert_game(_game())
    store.set_prediction("1", "fixture", {"p_H": 0.5, "p_D": 0.3, "p_A": 0.2})
    assert not store.has_lineup("1")
    store.set_lineup("1", [{"side": "home", "name": "A", "position": "G"}])
    store.set_lineup("1", [{"side": "away", "name": "B", "position": "G", "form90": 0.4}])
    store.set_prediction("1", "lineup", {"p_H": 0.4, "p_D": 0.3, "p_A": 0.3})
    g = store.game("1")
    assert g is not None
    assert (g["fixture_p_H"], g["lineup_p_H"]) == (0.5, 0.4)
    assert [p["name"] for p in g["lineup"]] == ["B"]  # replaced, not appended
    assert store.games(since="2026-10-18") == []
    assert store.game("nope") is None


# --- poller -----------------------------------------------------------------------


class FakePredictor(Predictor):
    """Fixtures and form in memory; predicts from the mean home form90 when there is one."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        stat, matches = _player_stat()
        self._fixtures = pl.DataFrame(
            {
                "game_id": [99],
                "date": [datetime(2026, 10, 17, 14)],
                "league": ["ENG-Premier League"],
                "home_team": ["chelsea"],
                "away_team": ["everton"],
            }
        )
        self._form = current_form(stat, matches)

    def fixtures(self) -> pl.DataFrame:
        """The one fixture: chelsea v everton, game 99."""
        return self._fixtures

    def features(self, game_id: int) -> pl.DataFrame | None:
        """Empty slots for game 99."""
        if game_id != 99:
            return None
        return pl.DataFrame({"game_id": [99]}).with_columns(
            pl.lit(None, pl.Float64).alias(c) for c in slot_columns()
        )

    def predict(self, frame: pl.DataFrame) -> dict[str, float]:
        """0.6 home with a line-up in the slots, else 0.5."""
        lineup = frame["h_player1_prev_min"][0] is not None
        return (
            {"p_H": 0.6, "p_D": 0.2, "p_A": 0.2}
            if lineup
            else {"p_H": 0.5, "p_D": 0.3, "p_A": 0.2}
        )

    def form(self) -> pl.DataFrame:
        """current_form of _player_stat."""
        return self._form


def _fetchers(state: str, calls: list[str]) -> Fetchers:
    def games_on(league: str, day: date) -> list[espn.Game]:
        if league != "ENG-Premier League":
            return []
        body = {"events": [_event("1", state, "Chelsea FC", "Everton")]}
        return espn.parse_scoreboard(body, league)

    def lineups(league: str, espn_id: str) -> list[espn.Starter]:
        calls.append(espn_id)
        home = ["Alexis Sánchez", "Harry Kane"] + [f"H{i}" for i in range(9)]
        return espn.parse_lineups(_summary(home, [f"A{i}" for i in range(11)]))

    return Fetchers(games_on=games_on, lineups=lineups)


def _settings(tmp_path: Path) -> Settings:
    return Settings(live=LiveSettings(db_file=tmp_path / "live.sqlite", lineup_window_minutes=90))


def test_find_fixture_matches_teams_and_game(tmp_path: Path) -> None:
    game = espn.parse_scoreboard(
        {"events": [_event("1", "pre", "Chelsea FC", "Everton")]}, "ENG-Premier League"
    )[0]
    ids = find_fixture(game, FakePredictor(_settings(tmp_path)).fixtures())
    assert ids == {"home_team": "chelsea", "away_team": "everton", "game_id": 99}
    assert find_fixture(game, None)["game_id"] is None


def test_refresh_waits_for_the_lineup_window(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, predictor, calls = Store(settings.live.db_file), FakePredictor(settings), []
    refresh(settings, store, predictor, _fetchers("pre", calls), now=KICKOFF - timedelta(hours=3))
    (g,) = store.games()
    assert g["game_id"] == 99 and g["fixture_p_H"] == 0.5 and g["lineup_p_H"] is None
    assert calls == []  # three hours out: no line-up request yet


def test_refresh_stores_the_lineup_and_repredicts_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, predictor, calls = Store(settings.live.db_file), FakePredictor(settings), []
    now = KICKOFF - timedelta(minutes=50)
    refresh(settings, store, predictor, _fetchers("pre", calls), now=now)
    refresh(settings, store, predictor, _fetchers("in", calls), now=now + timedelta(hours=1))
    g = store.game("1")
    assert g is not None
    assert (g["state"], g["fixture_p_H"], g["lineup_p_H"]) == ("in", 0.5, 0.6)
    assert calls == ["1"]  # stored after the first time, not fetched again
    home = [p for p in g["lineup"] if p["side"] == "home"]
    assert home[0]["player_id"] == 7 and home[0]["prev_min"] == 900
    assert len(g["lineup"]) == 22


def test_refresh_survives_a_failing_league(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def boom(league: str, day: date) -> list[espn.Game]:
        raise OSError("ESPN down")

    seen = refresh(
        settings, Store(settings.live.db_file), FakePredictor(settings), Fetchers(games_on=boom)
    )
    assert seen == 0


def test_predictor_without_files_or_with_bad_files_gives_none(tmp_path: Path) -> None:
    from money_printer_research.config import CleanSettings, ModelingSettings, UnderstatSettings

    settings = Settings(
        clean=CleanSettings(out_dir=tmp_path),
        modeling=ModelingSettings(out_file=tmp_path / "model.joblib"),
        understat=UnderstatSettings(schedule_file=tmp_path / "schedule.csv"),
    )
    predictor = Predictor(settings)
    assert predictor.load() is None  # nothing there yet
    for f in predictor._files():  # pyright: ignore[reportPrivateUsage]
        f.write_text("not what it should be")
    assert predictor.load() is None  # unreadable: logged, no predictions
    assert predictor.fixtures() is None and predictor.form() is None
    assert predictor.features(1) is None


def _short_xi(league: str, espn_id: str) -> list[espn.Starter]:
    return espn.parse_lineups(_summary([f"H{i}" for i in range(10)], [f"A{i}" for i in range(11)]))


def test_refresh_waits_while_an_xi_is_incomplete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, calls = Store(settings.live.db_file), []
    fetch = _fetchers("in", calls)
    fetch.lineups = _short_xi
    refresh(settings, store, FakePredictor(settings), fetch, now=KICKOFF)
    assert not store.has_lineup("1")  # ten home starters: asked again next poll


def test_refresh_stores_an_unmatched_game_without_predictions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = Store(settings.live.db_file)

    def games_on(league: str, day: date) -> list[espn.Game]:
        body = {"events": [_event("7", "in", "Real Madrid", "Barcelona", ("1", "0"))]}
        return espn.parse_scoreboard(body, league) if league == "ESP-La Liga" else []

    def lineups(league: str, espn_id: str) -> list[espn.Starter]:
        return espn.parse_lineups(
            _summary([f"H{i}" for i in range(11)], [f"A{i}" for i in range(11)])
        )

    refresh(settings, store, FakePredictor(settings), Fetchers(games_on, lineups), now=KICKOFF)
    game = store.game("7")
    assert game is not None
    assert (game["game_id"], game["fixture_p_H"], game["lineup_p_H"]) == (None, None, None)
    assert len(game["lineup"]) == 22 and all(p["player_id"] is None for p in game["lineup"])
