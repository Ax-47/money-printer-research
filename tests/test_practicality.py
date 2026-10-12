"""analysis.practicality: time to significance, account limits, the exchange."""

import datetime

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.practicality import (
    account_limits,
    exchange_value,
    time_to_significance,
)

DAY = datetime.date(2024, 8, 17)


def _placed(profits: list[float], seasons: int = 1) -> pl.DataFrame:
    n = len(profits)
    return pl.DataFrame(
        {
            "variant": "v",
            "season_start": [2020 + i * seasons // n for i in range(n)],
            "day": [DAY + datetime.timedelta(days=i) for i in range(n)],
            "profit": profits,
            "close_ev": 0.05,
        }
    )


def test_time_to_significance() -> None:
    profits = [1.0, -1.0] * 50 + [1.0] * 10  # mean 10/110, sd about 0.96
    row = time_to_significance(_placed(profits, seasons=2), ["v"]).row(0, named=True)
    sd = float(np.std(profits, ddof=1))
    assert row["bets_per_season"] == 55
    assert row["seasons_needed_roi"] == pytest.approx((2 * sd / (10 / 110)) ** 2 / 55)
    assert row["seasons_needed_close_ev"] == pytest.approx((2 * sd / 0.05) ** 2 / 55)
    assert row["units_per_season_close_ev"] == pytest.approx(0.05 * 55)


def test_account_limits_stop_when_every_account_is_closed() -> None:
    placed = _placed([1.0] * 10)
    table = account_limits(placed, "v", accounts=(2,), limits=(2.0,))
    unlimited, limited = table.row(0, named=True), table.row(1, named=True)
    assert unlimited["profit"] == 10
    # Two accounts, each closed once 2 up: 4 bets in all.
    assert limited["bets"] == 4
    assert limited["profit"] == 4


def _prices() -> pl.DataFrame:
    """One game: the exchange offers 2.2 on a home win Pinnacle makes a coin flip."""
    rows = []
    for source, timing, odds in [
        ("betfair_exchange", "early", [2.2, 3.6, 3.6]),
        ("pinnacle", "early", [1.95, 3.9, 3.9]),
        ("pinnacle", "close", [1.95, 3.9, 3.9]),
    ]:
        for selection, o in zip("HDA", odds, strict=True):
            rows.append(("L", 2024, DAY, "a", "b", "1x2", selection, source, timing, None, o))
    return pl.DataFrame(
        rows,
        schema={
            "league": pl.String,
            "season_start": pl.Int32,
            "day": pl.Date,
            "fd_home": pl.String,
            "fd_away": pl.String,
            "market": pl.String,
            "selection": pl.String,
            "source": pl.String,
            "timing": pl.String,
            "line": pl.Float64,
            "odds": pl.Float64,
        },
        orient="row",
    )


def test_exchange_commission_comes_off_the_winnings() -> None:
    results = pl.DataFrame(
        {
            "league": "L",
            "season_start": pl.Series([2024], dtype=pl.Int32),
            "day": DAY,
            "fd_home": "a",
            "fd_away": "b",
            "home_goals": pl.Series([1], dtype=pl.Int32),
            "away_goals": pl.Series([0], dtype=pl.Int32),
        }
    )
    table = exchange_value(_prices(), results, min_edge=0.02, commissions=(0.05, 0.2))
    low, high = table.row(0, named=True), table.row(1, named=True)
    # 5%: 2.2 pays 1.14 net; the fair home probability is about 0.5, so the bet is taken.
    assert low["bets"] == 1
    assert low["roi"] == pytest.approx(1.2 * 0.95)
    # 20%: 1.96 net is no longer value.
    assert high["bets"] == 0
