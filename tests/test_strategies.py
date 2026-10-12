"""analysis.strategies: settling bets, the strategy rules, and multiple comparisons."""

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.strategies import (
    MIN_BOOK,
    Strategy,
    _wide,  # pyright: ignore[reportPrivateUsage]
    benjamini_hochberg,
    bets,
    load_suite,
    settle,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]


def _settled(market: str, selection: str, line: float, score: tuple[int, int]) -> float:
    row = pl.DataFrame(
        {
            "selection": [selection],
            "line": [line],
            "odds": [2.0],
            "home_goals": [score[0]],
            "away_goals": [score[1]],
        }
    )
    return row.select(settle(market)).item()


@pytest.mark.parametrize(
    ("market", "selection", "line", "score", "profit"),
    [
        ("1x2", "H", 0.0, (2, 1), 1.0),
        ("1x2", "D", 0.0, (2, 1), -1.0),
        ("1x2", "D", 0.0, (1, 1), 1.0),
        ("ou25", "over", 2.5, (2, 1), 1.0),
        ("ou25", "under", 2.5, (2, 1), -1.0),
        # Home -1: a one-goal win is a push, two goals a win.
        ("ah", "home", -1.0, (2, 1), 0.0),
        ("ah", "home", -1.0, (3, 1), 1.0),
        ("ah", "away", -1.0, (2, 1), 0.0),
        # Home -0.25 in a draw: half pushes (line 0), half loses (line -0.5).
        ("ah", "home", -0.25, (1, 1), -0.5),
        ("ah", "away", -0.25, (1, 1), 0.5),
        # Home -0.75 winning by one: half wins (-0.5), half pushes (-1).
        ("ah", "home", -0.75, (1, 0), 0.5),
        ("ah", "home", 0.5, (0, 0), 1.0),
    ],
)
def test_settle(
    market: str, selection: str, line: float, score: tuple[int, int], profit: float
) -> None:
    assert _settled(market, selection, line, score) == pytest.approx(profit)


def _offers() -> pl.DataFrame:
    """Two 1X2 games, three selections each."""
    return pl.DataFrame(
        {
            "league": "L",
            "season_start": 2020,
            "day": [datetime.date(2020, 8, 1)] * 3 + [datetime.date(2020, 8, 2)] * 3,
            "fd_home": ["a"] * 3 + ["c"] * 3,
            "fd_away": ["b"] * 3 + ["d"] * 3,
            "selection": ["H", "D", "A"] * 2,
            "odds": [1.5, 4.0, 7.0, 2.6, 3.3, 2.9],
            "fair": [0.64, 0.24, 0.12, 0.37, 0.29, 0.34],
            "model_p": [0.60, 0.30, 0.10, 0.40, 0.25, 0.35],
        }
    )


def _rule(kind: str, value: object = None, selection: str | None = None) -> list[str]:
    strategy = Strategy("s", kind, "1x2", "pinnacle", "early", selection=selection)
    taken = bets(_offers(), strategy, value)
    return (taken["fd_home"] + taken["selection"]).to_list()


def test_rules() -> None:
    assert _rule("always", selection="D") == ["aD", "cD"]
    assert _rule("favourite") == ["aH", "cH"]
    assert _rule("longshot") == ["aA", "cD"]
    assert _rule("odds_band", [2.0, 3.0]) == ["cH", "cA"]
    # fair * odds - 1: a H -0.04, D -0.04, A -0.16; c H -0.04, D -0.04, A -0.01.
    assert _rule("value", -0.02) == ["cA"]
    # model_p * odds - 1: a D +0.2 is a's best; c H +0.04 is c's best.
    assert _rule("model_ev", 0.1) == ["aD"]
    assert _rule("model_ev", 0.0) == ["aD", "cH"]
    # model_p - fair on the draw: a +0.06, c -0.04.
    assert _rule("model_draw", 0.05) == ["aD"]


def test_benjamini_hochberg() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.5])
    # Sorted 0.01 0.03 0.04 0.5 -> 0.04 0.06 0.0533 0.5, then made monotone.
    assert benjamini_hochberg(p) == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5])


def test_walk_forward_chooses_from_earlier_seasons() -> None:
    placed = pl.DataFrame(
        {
            "strategy": "s",
            "league": "L",
            "value": ["low", "high"] * 3,
            "season_start": [2020, 2020, 2021, 2021, 2022, 2022],
            "profit": [-1.0, 1.0, 5.0, -1.0, -1.0, 1.0],
        }
    )
    row = walk_forward(placed).filter(pl.col("league") == "L").row(0, named=True)
    # 2021 follows 2020's best ("high": -1); 2022 follows 2020-21 ("low": 4 vs 0): -1.
    assert row["seasons"] == 2
    assert row["profit"] == pytest.approx(-2.0)


def test_the_repository_suite_loads() -> None:
    test_from, suite = load_suite(ROOT / "strategies.toml")
    assert test_from == 2016
    assert len({s.name for s in suite}) == len(suite)
    assert {s.kind for s in suite} == {
        "model_ev",
        "model_draw",
        "favourite",
        "longshot",
        "odds_band",
        "always",
        "value",
    }


def test_prices_with_an_impossible_arbitrage_are_dropped() -> None:
    def game(home: str, odds: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "league": "L",
                "season_start": pl.Series([2020] * 2, dtype=pl.Int32),
                "day": datetime.date(2020, 8, 1),
                "fd_home": home,
                "fd_away": "x",
                "market": "ah",
                "selection": ["home", "away"],
                "source": "max",
                "timing": "early",
                "line": -0.5,
                "odds": odds,
            }
        )

    # 1/2.1 + 1/22 = 0.52: a data error. 1/2.05 + 1/2.05 = 0.976: a small, real arbitrage.
    prices = pl.concat([game("typo", [2.1, 22.0]), game("fine", [2.05, 2.05])])
    assert 1 / 2.1 + 1 / 22 < MIN_BOOK < 2 / 2.05
    assert _wide(prices, "ah", "max", "early")["fd_home"].to_list() == ["fine"]
