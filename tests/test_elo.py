import datetime

import numpy as np
import polars as pl
import pytest
from pandera.typing.polars import DataFrame

from money_printer_research.exact_features.elo import (
    DEFAULT_ELO,
    PROMOTED_GAP,
    _goal_multiplier,  # pyright: ignore[reportPrivateUsage]
    get_elo,
    outcome_probs,
    score_grid,
    to_elo_input,
)
from money_printer_research.schema import EloInputCols


def _matches(rows: list[tuple[str, str, str, int, int]]) -> DataFrame[EloInputCols]:
    """EloInputCols from (season, home, away, home_goals, away_goals), one day apart."""
    start = datetime.date(2014, 8, 1)
    return DataFrame[EloInputCols](
        pl.DataFrame(
            {
                "season": [r[0] for r in rows],
                "match_date": [start + datetime.timedelta(days=i) for i in range(len(rows))],
                "home_team": [r[1] for r in rows],
                "away_team": [r[2] for r in rows],
                "full_time_home_goals": [r[3] for r in rows],
                "full_time_away_goals": [r[4] for r in rows],
                "full_time_result": [
                    "H" if r[3] > r[4] else "D" if r[3] == r[4] else "A" for r in rows
                ],
            }
        )
    )


@pytest.mark.parametrize(("gd", "expected"), [(0, 1.0), (1, 1.0), (-2, 1.5), (3, 1.75), (5, 2.0)])
def test_goal_multiplier(gd: int, expected: float) -> None:
    assert _goal_multiplier(gd) == expected


def test_to_elo_input_season_label_and_result() -> None:
    matches = pl.DataFrame(
        {
            "season_start": pl.Series([2014, 2019, 2099], dtype=pl.Int32),
            "date": [
                datetime.datetime(2014, 8, 16, 15),
                datetime.datetime(2019, 9, 1),
                datetime.datetime(2099, 9, 1),
            ],
            "home_team": ["a", "b", "c"],
            "away_team": ["b", "c", "a"],
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
        }
    )
    out = to_elo_input(matches).unwrap()  # pyright: ignore[reportArgumentType]
    assert out["season"].to_list() == ["2014/15", "2019/20", "2099/00"]
    assert out["full_time_result"].to_list() == ["H", "D", "A"]
    assert out["match_date"].to_list()[0] == datetime.date(2014, 8, 16)


def test_get_elo_is_pre_match_and_zero_sum() -> None:
    elo = get_elo(_matches([("2014/15", "a", "b", 3, 0), ("2014/15", "b", "a", 1, 1)]))
    first, second = elo.row(0, named=True), elo.row(1, named=True)

    # Everyone starts the first season at the promoted rating, so the opener is level.
    start = DEFAULT_ELO.base - PROMOTED_GAP
    assert first["elo_home"] == first["elo_away"] == start
    assert first["elo_diff"] == 0
    # The winner gains exactly what the loser drops, and it shows only in the next match.
    gain = second["elo_away"] - start
    assert gain > 0
    assert second["elo_home"] == pytest.approx(start - gain)


def test_get_elo_promoted_team_inherits_relegated_rating() -> None:
    elo = get_elo(_matches([("2014/15", "a", "b", 4, 0), ("2015/16", "a", "c", 0, 0)]))
    opener, next_season = elo.row(0, named=True), elo.row(1, named=True)
    # Ratings are zero-sum, so b ends 2014/15 down by what a gained.
    a_gain = next_season["elo_home"] - opener["elo_home"]
    b_end = opener["elo_away"] - a_gain
    # b went down, so the promoted c starts 2015/16 at b's end rating.
    assert next_season["elo_away"] == pytest.approx(b_end)


def test_score_grid_and_outcomes_are_probabilities() -> None:
    lam, mu = np.array([1.5, 0.4]), np.array([1.1, 2.3])
    for rho in (0.0, -0.1):
        grid = score_grid(lam, mu, rho)
        assert grid.shape == (2, 11, 11)
        np.testing.assert_allclose(grid.sum(axis=(1, 2)), 1.0)
        probs = outcome_probs(grid)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()


def test_equal_rates_give_equal_win_chances() -> None:
    probs = outcome_probs(score_grid(np.array([1.3]), np.array([1.3])))[0]
    assert probs[0] == pytest.approx(probs[2])


def test_negative_rho_raises_draws() -> None:
    lam, mu = np.array([1.3]), np.array([1.1])
    draw_plain = outcome_probs(score_grid(lam, mu, 0.0))[0, 1]
    draw_dc = outcome_probs(score_grid(lam, mu, -0.1))[0, 1]
    assert draw_dc > draw_plain
