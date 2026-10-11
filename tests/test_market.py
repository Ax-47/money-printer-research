"""analysis.market: removing the margin, team form, and comparing forecasters."""

import datetime

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.market import (
    FORECASTERS,
    FORM_COLUMNS,
    forecasts,
    losses,
    margin_free,
    paired_differences,
    team_form,
)

ODDS = np.array([[1.5, 4.2, 7.0], [2.6, 3.3, 2.9], [1.1, 11.0, 26.0]])


@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_margin_free_probabilities_sum_to_one(method: str) -> None:
    p = margin_free(ODDS, method)
    assert p.sum(axis=1) == pytest.approx(np.ones(3))
    # Removing the margin lowers every implied probability, keeping their order.
    assert (p < 1 / ODDS).all()
    assert (np.argsort(p, axis=1) == np.argsort(1 / ODDS, axis=1)).all()


@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_fair_odds_are_left_alone(method: str) -> None:
    fair = np.array([[2.0, 4.0, 4.0]])
    assert margin_free(fair, method) == pytest.approx(np.array([[0.5, 0.25, 0.25]]))


def test_power_takes_more_margin_off_the_long_shot() -> None:
    proportional, power = margin_free(ODDS, "proportional"), margin_free(ODDS, "power")
    assert power[2, 2] < proportional[2, 2]  # the 26.0 outsider
    assert power[2, 0] > proportional[2, 0]  # the 1.1 favourite


def _games(n: int, seed: int = 0) -> pl.DataFrame:
    """Make n games between four teams, results drawn from the early odds' probabilities."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    home = rng.choice(teams, n)
    away = np.array([rng.choice([t for t in teams if t != h]) for h in home])
    p = rng.dirichlet([5, 3, 3], n)
    result = (rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1)
    hg = np.where(result == 0, 2, np.where(result == 1, 1, 0))
    ag = np.where(result == 2, 2, np.where(result == 1, 1, 0))
    early = 0.95 / p
    return pl.DataFrame(
        {
            "league": "L",
            "season_start": np.repeat(np.arange(2014, 2014 + n // 100 + 1), 100)[:n],
            "day": [datetime.date(2014, 8, 1) + datetime.timedelta(days=i) for i in range(n)],
            "fd_home": home,
            "fd_away": away,
            "home_goals": hg,
            "away_goals": ag,
            "home_shots_on_target": rng.integers(0, 10, n),
            "away_shots_on_target": [None, *rng.integers(0, 10, n - 1)],
            "result": np.array(["H", "D", "A"])[result],
            **{f"early_{k}": early[:, i] for i, k in enumerate("HDA")},
            **{f"close_{k}": early[:, i] * 0.99 for i, k in enumerate("HDA")},
        }
    )


def test_team_form_uses_only_earlier_games() -> None:
    games = _games(3)
    form = team_form(games)
    # Nobody has played before the first game: equal form.
    assert form.row(0, named=True)["elo_diff"] == 0
    # Changing the last game's score cannot change its own form.
    changed = games.with_columns(
        home_goals=pl.when(pl.int_range(3) == 2).then(9).otherwise(pl.col("home_goals"))
    )
    assert team_form(changed).select(FORM_COLUMNS).row(2) == form.select(FORM_COLUMNS).row(2)


def test_every_forecaster_is_scored_on_the_same_test_games() -> None:
    table = losses(forecasts(_games(400), first_test_season=2016))
    counts = table.group_by("forecaster").len()
    assert sorted(counts["forecaster"].to_list()) == sorted(FORECASTERS)
    assert counts["len"].unique().to_list() == [200]  # seasons 2016 and 2017


def test_paired_differences_are_a_minus_b() -> None:
    table = pl.DataFrame(
        {
            "forecaster": ["a"] * 4 + ["b"] * 4,
            "game": [0, 1, 2, 3] * 2,
            "league": "L",
            "season_start": 2020,
            "log_loss": [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.2],
            "rps": [0.2] * 8,
        }
    )
    row = paired_differences(table, [("a", "b")]).row(0, named=True)
    assert row["games"] == 4
    assert row["log_loss"] == pytest.approx(-0.15)
    assert row["log_loss_low"] < row["log_loss"] < row["log_loss_high"] < 0
    assert row["rps"] == pytest.approx(0.0)
