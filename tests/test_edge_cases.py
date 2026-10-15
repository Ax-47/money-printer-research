"""Branches the main tests do not reach: unknown names, missing columns, empty sections."""

import urllib.error
from email.message import Message
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from money_printer_research import config
from money_printer_research.analysis import report, staking, strategies
from money_printer_research.analysis.odds import read_prices
from money_printer_research.data_collection import scrape_odds
from money_printer_research.features.elo import get_elo, walk_forward
from tests.test_elo import _matches  # pyright: ignore[reportPrivateUsage]
from tests.test_report import _results  # pyright: ignore[reportPrivateUsage]


def test_num_and_pct_show_a_dash_for_none() -> None:
    assert report._num(None) == "–"  # pyright: ignore[reportPrivateUsage]
    assert report._pct(None) == "–"  # pyright: ignore[reportPrivateUsage]


def test_summary_leaves_out_claims_without_numbers() -> None:
    r = _results()
    r["scores"] = r["scores"].filter(~pl.col("variant").str.starts_with("value"))
    r["staking"]["ruin"] = r["staking"]["ruin"].filter(pl.col("plan") != "martingale")
    headlines = [h for h, _ in report.summary(r)]
    full = [h for h, _ in report.summary(_results())]
    assert len(headlines) == 2 < len(full)


def test_implausible_prices_skips_a_price_missing_a_selection() -> None:
    prices = pl.DataFrame(
        {
            "league": ["L"] * 2,
            "season_start": [2020] * 2,
            "day": [None, None],
            "fd_home": ["a", "a"],
            "fd_away": ["b", "b"],
            "market": ["1x2", "1x2"],
            "selection": ["H", "D"],  # no away price
            "source": ["x", "x"],
            "timing": ["early", "early"],
            "line": [None, None],
            "odds": [2.0, 3.0],
        }
    )
    assert strategies.implausible_prices(prices).is_empty()


def test_unknown_strategy_kind_and_staking_plan_raise() -> None:
    strategy = strategies.Strategy("x", "nonsense", "1x2", "pinnacle", "early")
    with pytest.raises(ValueError, match="unknown strategy kind"):
        strategies.bets(pl.DataFrame({"odds": [2.0]}), strategy, None)
    ones = np.ones((1, 1))
    with pytest.raises(ValueError, match="unknown staking plan"):
        staking.simulate(ones, ones * 2, ones / 2, "nonsense")


def test_no_settings_file_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "__file__", str(tmp_path / "pkg" / "config.py"))
    assert config.find_settings_file(env={}, cwd=tmp_path) is None


def test_download_reraises_other_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(url: str, path: Path) -> None:
        raise urllib.error.HTTPError(url, 500, "Server Error", Message(), None)

    monkeypatch.setattr(scrape_odds.urllib.request, "urlretrieve", fail)
    with pytest.raises(urllib.error.HTTPError):
        scrape_odds._download("https://example.invalid/x.csv", tmp_path / "x.csv")  # pyright: ignore[reportPrivateUsage]


def test_walk_forward_without_dixon_coles() -> None:
    rows = [
        (season, h, a, hg, ag)
        for season in ("2014/15", "2015/16")
        for (h, a), (hg, ag) in zip(
            [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")],
            [(2, 0), (1, 1), (3, 1), (0, 0), (1, 2), (2, 2)],
            strict=True,
        )
    ]
    pred = walk_forward(get_elo(_matches(rows)), ["2015/16"], dixon_coles=False)
    assert pred["rho"].unique().to_list() == [0.0]
    total = pred.select(pl.sum_horizontal("p_home", "p_draw", "p_away"))
    assert np.allclose(total.to_series().to_numpy(), 1.0)


def test_asian_handicap_prices_without_a_line_are_skipped(tmp_path: Path) -> None:
    league = tmp_path / "GER-Bundesliga"
    league.mkdir()
    pl.DataFrame(
        {
            "Div": ["D1"],
            "Date": ["23/08/2024"],
            "HomeTeam": ["Bayern Munich"],
            "AwayTeam": ["Wolfsburg"],
            "FTHG": ["3"],
            "FTAG": ["2"],
            "PAHH": ["1.95"],  # no AHh column to give the line
            "PAHA": ["1.93"],
            "B365H": ["1.22"],
        }
    ).write_csv(league / "2024-25.csv")
    prices = read_prices(tmp_path)
    assert set(prices["market"]) == {"1x2"}
