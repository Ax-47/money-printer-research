import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.backtest import (
    attach_odds,
    cumulative_profit,
    log_loss_vs_market,
    odds_url,
    picks,
    read_odds,
    summary,
    team_names,
)

LEAGUE = "ENG-Premier League"
DAY = datetime.date(2024, 8, 17)


NAMES = {
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "everton": "Everton",
    "manchester united": "Man United",
}
ROUNDS = [
    [("arsenal", "chelsea"), ("everton", "manchester united")],
    [("chelsea", "everton"), ("manchester united", "arsenal")],
    [("arsenal", "everton"), ("chelsea", "manchester united")],
    [("chelsea", "arsenal"), ("manchester united", "everton")],
    [("everton", "chelsea"), ("arsenal", "manchester united")],
    [("everton", "arsenal"), ("manchester united", "chelsea")],
]
"""A double round robin of four teams, one round a week, in match_features naming."""


def _games() -> pl.DataFrame:
    rows = [
        (round_ * 7, home, away) for round_, games in enumerate(ROUNDS) for home, away in games
    ]
    return pl.DataFrame(
        {
            "game_id": range(1, len(rows) + 1),
            "date": [
                datetime.datetime.combine(DAY, datetime.time(15)) + datetime.timedelta(days=r[0])
                for r in rows
            ],
            "league": LEAGUE,
            "season_start": 2024,
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "full_time_result": "H",
        }
    )


def _write_odds(odds_dir: Path) -> None:
    """The same games in football-data's E0 CSV, its team names and date format.

    The first game has no Pinnacle price.
    """
    games = _games()
    odds_dir.mkdir()
    pinnacle = [None, *[float(i) for i in range(2, games.height + 1)]]
    pl.DataFrame(
        {
            "Div": "E0",
            "Date": games["date"].dt.strftime("%d/%m/%Y"),
            "HomeTeam": games["home_team"].replace_strict(NAMES),
            "AwayTeam": games["away_team"].replace_strict(NAMES),
            "PSCH": pinnacle,
            "PSCD": 3.5,
            "PSCA": pinnacle,
            "B365H": 1.5,
            "B365D": 3.4,
            "B365A": 6.0,
        }
    ).write_csv(odds_dir / "E0_2024.csv")


def test_odds_url() -> None:
    assert odds_url("E0", 2024) == "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    assert odds_url("SP1", 2009) == "https://www.football-data.co.uk/mmz4281/0910/SP1.csv"


def test_read_odds_parses_dates_and_falls_back_to_other_odds(tmp_path: Path) -> None:
    _write_odds(tmp_path / "odds")
    odds = read_odds(tmp_path / "odds")
    assert odds["day"].to_list() == _games()["date"].dt.date().to_list()
    assert odds["odds_H"].to_list()[:3] == [1.5, 2.0, 3.0]  # no Pinnacle for game 1: Bet365


def test_team_names_are_learned_from_shared_days(tmp_path: Path) -> None:
    _write_odds(tmp_path / "odds")
    names = team_names(read_odds(tmp_path / "odds"), _games())
    mapping = dict(zip(names["fd_name"], names["team"], strict=True))
    assert mapping == {fd: team for team, fd in NAMES.items()}


def test_attach_odds_matches_every_game(tmp_path: Path) -> None:
    _write_odds(tmp_path / "odds")
    games = attach_odds(_games(), read_odds(tmp_path / "odds")).sort("game_id")
    assert games["game_id"].to_list() == list(range(1, 13))
    assert games["odds_A"].to_list() == [6.0, *[float(i) for i in range(2, 13)]]


def _table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [datetime.datetime(2024, 8, d) for d in (17, 17, 24)],
            "season_start": 2024,
            "full_time_result": ["H", "D", "A"],
            "p_H": [0.6, 0.3, 0.5],
            "p_D": [0.2, 0.3, 0.3],
            "p_A": [0.2, 0.4, 0.2],
            "odds_H": [2.0, 3.0, 1.8],
            "odds_D": [3.5, 3.2, 3.6],
            "odds_A": [4.0, 2.5, 4.5],
        }
    )


def test_picks_take_the_best_ev_and_settle_it() -> None:
    table = picks(_table())
    # EV per game, H D A: 0.2 -0.3 -0.2; -0.1 -0.04 0; -0.1 0.08 -0.1.
    assert table["pick"].to_list() == ["H", "A", "D"]
    assert table["ev"].to_list() == pytest.approx([0.2, 0.0, 0.08])
    assert table["profit"].to_list() == pytest.approx([1.0, -1.0, -1.0])


def test_summary_and_cumulative_profit_filter_on_ev() -> None:
    table = picks(_table())
    total = summary(table, 0.05).row(0, named=True)
    assert total["bets"] == 2
    assert total["profit"] == pytest.approx(0.0)
    assert total["hit_rate"] == pytest.approx(0.5)
    curve = cumulative_profit(table, 0.05)
    assert curve["profit"].to_list() == pytest.approx([1.0, 0.0])
    assert summary(table, 1.0).row(0, named=True)["bets"] == 0


def test_market_log_loss_removes_the_margin() -> None:
    table = _table().with_columns(odds_H=pl.lit(2.0), odds_D=pl.lit(4.0), odds_A=pl.lit(4.0))
    # 1/2 + 1/4 + 1/4 = 1, so no margin to remove: the market says H 0.5, D 0.25, A 0.25.
    losses = log_loss_vs_market(table)
    assert losses["market"] == pytest.approx(-(np.log(0.5) + 2 * np.log(0.25)) / 3)
    assert losses["model"] == pytest.approx(-(np.log(0.6) + np.log(0.3) + np.log(0.2)) / 3)
