import datetime

import polars as pl
import pytest
from pandera.typing.polars import DataFrame

from money_printer_research.exact_features.fatigue import player_fatigue, team_fatigue
from money_printer_research.schema import FatigueInputCols

DAY0 = datetime.date(2020, 1, 1)


def _player_matches(
    rows: list[tuple[int, int, int, int, str, bool, int]],
) -> DataFrame[FatigueInputCols]:
    """FatigueInputCols from (game_id, day, player_id, team_id, position, is_home, minutes)."""
    n = len(rows)
    return DataFrame[FatigueInputCols](
        pl.DataFrame(
            {
                "league": ["ENG-Premier League"] * n,
                "season": ["1920"] * n,
                "game": ["g"] * n,
                "team": ["t"] * n,
                "player": [f"p{r[2]}" for r in rows],
                "game_id": [r[0] for r in rows],
                "team_id": [r[3] for r in rows],
                "player_id": [r[2] for r in rows],
                "position": [r[4] for r in rows],
                "minutes": [r[6] for r in rows],
                "xg": [0.0] * n,
                "xa": [0.0] * n,
                "goals": [0] * n,
                "own_goals": [0] * n,
                "shots": [0] * n,
                "xg_chain": [0.0] * n,
                "xg_buildup": [0.0] * n,
                "assists": [0] * n,
                "key_passes": [0] * n,
                "yellow_cards": [0] * n,
                "red_cards": [0] * n,
                "season_start": pl.Series([2019] * n, dtype=pl.Int32),
                "match_date": [DAY0 + datetime.timedelta(days=r[1]) for r in rows],
                "is_home": [r[5] for r in rows],
            }
        )
    )


def test_player_fatigue_load_uses_only_earlier_matches() -> None:
    # Player 1 plays team 10's games on days 0, 3 and 6; player 2 skips the day-3 game.
    pm = _player_matches(
        [
            (1, 0, 1, 10, "MC", True, 90),
            (2, 3, 1, 10, "MC", False, 90),
            (3, 6, 1, 10, "MC", True, 90),
            (1, 0, 2, 10, "FW", True, 90),
            (3, 6, 2, 10, "FW", True, 90),
        ]
    )
    pf = player_fatigue(pm).sort("player_id", "match_date")
    p1 = pf.filter(pl.col("player_id") == 1)
    assert p1["rest_days"].to_list() == [None, 3, 3]
    assert p1["min_7d"].to_list() == [0, 90, 180]
    assert p1["n_7d"].to_list() == [0, 1, 2]
    assert p1["streak_prev"].to_list() == [0, 1, 2]
    # Missing the team's day-3 game breaks player 2's streak.
    p2 = pf.filter(pl.col("player_id") == 2)
    assert p2["streak_prev"].to_list() == [0, 0]
    assert pf["fatigue_score"].is_between(0, 100).all()
    # A rested debut has no load at all.
    assert p1["fatigue_score"][0] == 0


def test_team_fatigue_averages_starters_per_side() -> None:
    pm = _player_matches(
        [
            (1, 0, 1, 10, "MC", True, 90),
            (1, 0, 2, 10, "Sub", True, 10),
            (1, 0, 3, 20, "GK", False, 90),
        ]
    )
    tf = team_fatigue(player_fatigue(pm))
    row = tf.row(0, named=True)
    assert tf.height == 1
    assert row["h_fatigue"] == pytest.approx(0)
    assert row["fatigue_diff"] == pytest.approx(row["h_fatigue"] - row["a_fatigue"])


def test_team_fatigue_side_without_starters_is_null() -> None:
    pm = _player_matches([(1, 0, 1, 10, "MC", True, 90), (1, 0, 2, 20, "Sub", False, 20)])
    row = team_fatigue(player_fatigue(pm)).row(0, named=True)
    assert row["a_fatigue"] is None
    assert row["a_min_7d"] is None
    assert row["fatigue_diff"] is None
