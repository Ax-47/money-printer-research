import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
from pandera.typing.polars import DataFrame

from money_printer_research.analysis.modeling import fit_final
from money_printer_research.analysis.predict import (
    fixture_features,
    main,
    games_in_window,
    last_starters,
    predict_fixtures,
)
from money_printer_research.clean_data.match_stat import clean_upcoming
from money_printer_research.config import Settings, UnderstatSettings
from money_printer_research.features.elo import get_elo, to_elo_input
from money_printer_research.schema import slot_columns
from money_printer_research.schema.match.raw import UnderstatScheduleRaw
from tests.test_integrity import _raw_schedule  # pyright: ignore[reportPrivateUsage]

TEAMS = ["a", "b", "c", "d"]
ID_COLS = ["game_id", "date", "home_team", "away_team"]
START = datetime.datetime(2024, 8, 3, 15)


def _history(weeks: int = 30, seed: int = 0) -> pl.DataFrame:
    """Played matches in match_features form: two games a week, `a` the strongest team."""
    rng = np.random.default_rng(seed)
    pairs = [("a", "b"), ("c", "d"), ("b", "c"), ("d", "a"), ("a", "c"), ("b", "d")]
    rows = []
    for i in range(weeks * 2):
        home, away = pairs[i % len(pairs)]
        strength = {"a": 1.5, "b": 1.0, "c": 1.0, "d": 0.7}
        rows.append(
            {
                "game_id": i,
                "date": START + datetime.timedelta(days=3.5 * i),
                "league": "ENG-Premier League",
                "season_start": 2024,
                "home_team": home,
                "away_team": away,
                "home_goals": int(rng.poisson(strength[home] * 1.2)),
                "away_goals": int(rng.poisson(strength[away])),
            }
        )
    df = pl.DataFrame(rows)
    slots = {c: rng.normal(size=df.height) for c in slot_columns()}
    result = (
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") == pl.col("away_goals"))
        .then(pl.lit("D"))
        .otherwise(pl.lit("A"))
    )
    # df is in date order already, so get_elo's rows line up with it.
    elo = get_elo(to_elo_input(df).unwrap())  # pyright: ignore[reportArgumentType]
    return df.with_columns(
        elo.select("elo_home", "elo_away", "elo_diff"),
        full_time_result=result,
        h_games_14d_all=pl.lit(2.0),
        a_games_14d_all=pl.lit(2.0),
        **{c: pl.Series(v) for c, v in slots.items()},
    )


def _fixtures(history: pl.DataFrame, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    """Unplayed fixtures one day apart, starting a week after the last played match."""
    first = history["date"].max() + datetime.timedelta(days=7)  # pyright: ignore[reportOperatorIssue]
    return pl.DataFrame(
        {
            "game_id": [1000 + i for i in range(len(pairs))],
            "date": [first + datetime.timedelta(days=i) for i in range(len(pairs))],
            "league": "ENG-Premier League",
            "season_start": 2024,
            "home_team": [h for h, _ in pairs],
            "away_team": [a for _, a in pairs],
        }
    )


def test_clean_upcoming_keeps_only_the_unplayed_fixture() -> None:
    raw = cast(DataFrame[UnderstatScheduleRaw], _raw_schedule())
    fixtures = clean_upcoming(raw).unwrap()
    assert fixtures["game_id"].to_list() == [103]
    assert fixtures["season_start"].to_list() == [2023]
    assert fixtures["home_goals"].to_list() == [None]


def test_fixture_elo_is_the_rating_after_the_last_played_match() -> None:
    history = _history()
    fixtures = _fixtures(history, [("a", "d"), ("b", "c")])
    elo = fixture_features(history, fixtures).sort("game_id")
    # The same fixtures played (any score): their pre-match ratings are the ones to match.
    played_too = pl.concat(
        [history, fixtures.with_columns(home_goals=pl.lit(1), away_goals=pl.lit(1))],
        how="diagonal_relaxed",
    )
    expected = (
        get_elo(to_elo_input(played_too).unwrap())  # pyright: ignore[reportArgumentType]
        .tail(2)
        .select("elo_home", "elo_away", "elo_diff")
    )
    assert elo.select("elo_home", "elo_away", "elo_diff").equals(expected)
    assert elo["elo_home"][0] > elo["elo_away"][0]  # a is stronger than d


def test_fixture_elo_starts_a_new_season_for_a_promoted_team() -> None:
    history = _history()
    now = fixture_features(history, _fixtures(history, [("b", "c"), ("d", "a")]))
    rating = dict(zip(now["home_team"], now["elo_home"])) | dict(
        zip(now["away_team"], now["elo_away"])
    )
    fixtures = _fixtures(history, [("e", "a")]).with_columns(season_start=pl.lit(2025))
    elo = fixture_features(history, fixtures)
    # e is new in 2025 and b, c, d are gone: e starts at their mean rating, a keeps its own.
    assert elo["elo_home"].item() == pytest.approx(np.mean([rating[t] for t in "bcd"]))
    assert elo["elo_away"].item() == pytest.approx(rating["a"])


def test_games_in_window_counts_league_games_in_the_14_days_before() -> None:
    history = _history(weeks=4)
    fixtures = _fixtures(history, [("a", "b"), ("a", "c")])
    counts = games_in_window(history, fixtures).sort("game_id")
    games = pl.concat([history.select(ID_COLS), fixtures.select(ID_COLS)])
    day = pl.col("date").dt.date()

    def expected(team: str, kickoff: datetime.datetime) -> int:
        own = games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
        start = kickoff.date() - datetime.timedelta(days=14)
        return own.filter((day < kickoff.date()) & (day >= start)).height

    kickoffs = fixtures["date"].to_list()
    assert counts["h_games_14d_all"].to_list() == [expected("a", k) for k in kickoffs]
    assert counts["a_games_14d_all"].to_list() == [
        expected(t, k) for t, k in zip("bc", kickoffs, strict=True)
    ]
    # The second fixture counts the first: a fixture is a game too.
    assert counts["h_games_14d_all"][1] == counts["h_games_14d_all"][0] + 1


def test_last_starters_takes_the_side_the_team_played_on() -> None:
    history = _history(weeks=3)
    starters = last_starters(history)
    assert sorted(starters["team"].to_list()) == TEAMS
    last_d = history.filter((pl.col("home_team") == "d") | (pl.col("away_team") == "d")).tail(1)
    side = "h" if last_d["home_team"].item() == "d" else "a"
    row = starters.filter(pl.col("team") == "d")
    assert row["player1_form90"].item() == last_d[f"{side}_player1_form90"].item()


def test_predict_fixtures_gives_probabilities_in_kickoff_order() -> None:
    history = _history()
    model = fit_final(history, ["elo"])
    fixtures = _fixtures(history, [("b", "c"), ("a", "d"), ("d", "a")]).with_columns(
        pl.col("date").reverse()
    )
    out = predict_fixtures(model, history, fixtures)
    assert out.columns == [
        "game_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "p_H",
        "p_D",
        "p_A",
    ]
    assert out["date"].is_sorted()
    assert out.select(pl.sum_horizontal("p_H", "p_D", "p_A")).to_series().to_list() == (
        pytest.approx([1.0] * 3)
    )
    a_home = out.filter(pl.col("home_team") == "a")
    assert a_home["p_H"].item() > a_home["p_A"].item()


def test_main_says_what_to_do_when_every_fixture_is_played(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schedule = tmp_path / "xg_schedule.csv"
    _raw_schedule().filter(pl.col("is_result")).write_csv(schedule)
    settings = Settings(understat=UnderstatSettings(schedule_file=schedule))
    assert main(settings).is_empty()
    assert "UNDERSTAT__END_YEAR" in capsys.readouterr().out
