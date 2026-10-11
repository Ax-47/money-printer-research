"""analysis.odds: football-data's price columns into one long table."""

from pathlib import Path

import polars as pl
import pytest

from money_printer_research.analysis.odds import one_price, parse_column, read_prices, read_results

LEAGUE = "GER-Bundesliga"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("PSCH", ("1x2", "H", "pinnacle", "close")),
        ("PSD", ("1x2", "D", "pinnacle", "early")),
        ("PAHH", ("ah", "home", "pinnacle", "early")),
        ("PC<2.5", ("ou25", "under", "pinnacle", "close")),
        ("B365>2.5", ("ou25", "over", "bet365", "early")),
        ("MaxCAHA", ("ah", "away", "max", "close")),
        ("BbMxA", ("1x2", "A", "max", "early")),
        ("BbAvAHH", ("ah", "home", "avg", "early")),
        ("BFEA", ("1x2", "A", "betfair_exchange", "early")),
        # 2024/25 has Betfair's sportsbook as BF; 2025/26 adds Betfred as BFD.
        ("BFD", ("1x2", "D", "betfair_sportsbook", "early")),
        ("BFDCH", ("1x2", "H", "betfred", "close")),
    ],
)
def test_parse_column(column: str, expected: tuple[str, str, str, str]) -> None:
    price = parse_column(column)
    assert price is not None
    assert (price["market"], price["selection"], price["source"], price["timing"]) == expected


@pytest.mark.parametrize("column", ["HS", "FTHG", "AHh", "BbAHh", "Referee", "PH", "PSAHH"])
def test_parse_column_skips_what_is_not_a_price(column: str) -> None:
    assert parse_column(column) is None


def _write(odds_dir: Path) -> None:
    """Two games in one football-data file: Pinnacle 1X2 for one, Bet365 for both."""
    (odds_dir / LEAGUE).mkdir(parents=True)
    pl.DataFrame(
        {
            "Div": ["D1", "D1"],
            "Date": ["23/08/2024", "24/08/24"],
            "HomeTeam": ["Bayern Munich", "Dortmund"],
            "AwayTeam": ["Wolfsburg", "Frankfurt"],
            "FTHG": ["3", "2"],
            "FTAG": ["2", "0"],
            "PSH": ["1.25", ""],
            "PSD": ["7.0", ""],
            "PSA": ["11.0", ""],
            "B365H": ["1.22", "1.8"],
            "B365D": ["6.5", "3.9"],
            "B365A": ["10.0", "4.2"],
            "AHh": ["-2.0", "-0.5"],
            "PAHH": ["1.95", "1.9"],
            "PAHA": ["1.93", "2.0"],
            "P>2.5": ["1.4", "1.6"],
            "P<2.5": ["3.0", "2.4"],
        }
    ).write_csv(odds_dir / LEAGUE / "2024-25.csv")


def test_read_prices_keeps_every_price_with_its_line(tmp_path: Path) -> None:
    _write(tmp_path)
    prices = read_prices(tmp_path)
    # 3 Pinnacle 1X2 for game 1, 3+3 Bet365 1X2, 2+2 AH, 2+2 O/U; empty cells dropped.
    assert prices.height == 3 + 6 + 4 + 4
    ah = prices.filter(pl.col("market") == "ah").sort("fd_home", "selection")
    assert ah["line"].to_list() == [-2.0, -2.0, -0.5, -0.5]
    assert prices.filter(pl.col("market") == "ou25")["line"].unique().to_list() == [2.5]
    assert prices["season_start"].unique().to_list() == [2024]


def test_read_results(tmp_path: Path) -> None:
    _write(tmp_path)
    results = read_results(tmp_path).sort("fd_home")
    assert results["home_goals"].to_list() == [3, 2]
    assert results["day"].dt.day().to_list() == [23, 24]


def test_one_price_takes_the_first_source_that_prices_every_outcome(tmp_path: Path) -> None:
    _write(tmp_path)
    wide = one_price(
        read_prices(tmp_path), "1x2", [("pinnacle", "early"), ("bet365", "early")]
    ).sort("fd_home")
    # Bayern has Pinnacle; Dortmund only Bet365.
    assert wide["odds_H"].to_list() == [1.25, 1.8]
    assert wide["odds_A"].to_list() == [11.0, 4.2]
