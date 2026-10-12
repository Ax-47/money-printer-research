"""analysis.staking: replaying bets under each staking plan."""

import datetime

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.staking import (
    FLAT_STAKE,
    PLANS,
    replay,
    ruin_probabilities,
    simulate,
)


def _path(returns: list[float], plan: str, odds: float = 2.0, p: float = 0.6) -> list[float]:
    r = np.array(returns)[:, None]
    n = len(returns)
    path = simulate(r, np.full((n, 1), odds), np.full((n, 1), p), plan)
    return path[:, 0].tolist()


def test_flat_stakes_the_same_every_time() -> None:
    assert _path([1.0, -1.0, -1.0], "flat") == pytest.approx(
        [1, 1 + FLAT_STAKE, 1, 1 - FLAT_STAKE]
    )


def test_kelly_stakes_its_fraction_of_the_current_bankroll() -> None:
    # p 0.6 at evens: Kelly stakes (0.6 * 2 - 1) / (2 - 1) = 20%.
    assert _path([1.0, -1.0], "kelly") == pytest.approx([1, 1.2, 0.96])
    assert _path([1.0], "half_kelly") == pytest.approx([1, 1.1])


def test_kelly_does_not_bet_without_an_edge() -> None:
    assert _path([-1.0, -1.0], "kelly", p=0.4) == [1, 1, 1]


def test_martingale_doubles_after_each_loss_and_resets_after_a_win() -> None:
    s = FLAT_STAKE
    assert _path([-1.0, -1.0, 1.0, -1.0], "martingale") == pytest.approx(
        [1, 1 - s, 1 - 3 * s, 1 + s, 1]
    )


def test_a_ruined_bankroll_stops_betting() -> None:
    # Kelly at p 0.999 stakes 99.8%; losing it leaves 0.002, under RUINED: no more bets.
    path = _path([-1.0, 1.0, 1.0], "kelly", p=0.999)
    assert path == pytest.approx([1, 0.002, 0.002, 0.002])


def _bets(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    won = rng.random(n) < 0.5
    return pl.DataFrame(
        {
            "day": [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)],
            "season_start": [2020 + i * 2 // n for i in range(n)],
            "odds": 1.95,
            "profit": np.where(won, 0.95, -1.0),
            "believed_p": 0.5,
        }
    )


def test_replay_has_every_plan() -> None:
    table = replay(_bets(50))
    assert table["plan"].to_list() == PLANS
    assert (table["max_drawdown"] >= 0).all()


def test_a_negative_edge_loses_more_often_over_more_seasons() -> None:
    ruin = ruin_probabilities(_bets(400), horizons=(1, 5), histories=400)
    flat = ruin.filter(pl.col("plan") == "flat").sort("seasons")
    assert flat["seasons"].to_list() == [1, 5]
    assert flat["median_final"][1] < flat["median_final"][0] < 1
    martingale = ruin.filter(pl.col("plan") == "martingale")
    assert (martingale["p_lose_half"] >= flat["p_lose_half"]).all()
