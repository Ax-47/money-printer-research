import datetime
from typing import cast

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.features.calendar import MAX_GAP_DAYS, club_calendar
from money_printer_research.schema import MatchLeaguesCols


def _games(rows: list[tuple[str, int, str, int, int]]) -> DataFrame[MatchLeaguesCols]:
    """The std_tm columns club_calendar reads, from (tm_game_id, day, comp_type, home, away).

    Only those columns, so the frame is cast rather than validated against MatchLeaguesCols.
    """
    day0 = datetime.date(2020, 1, 1)
    frame = pl.DataFrame(
        {
            "tm_game_id": [r[0] for r in rows],
            "date": [day0 + datetime.timedelta(days=r[1]) for r in rows],
            "comp_type": [r[2] for r in rows],
            "home_club_id": pl.Series([r[3] for r in rows], dtype=pl.Int32),
            "away_club_id": pl.Series([r[4] for r in rows], dtype=pl.Int32),
        }
    )
    return cast(DataFrame[MatchLeaguesCols], frame)


def test_club_calendar_counts_every_competition() -> None:
    # Club 1: league at home day 0, away in Europe day 3, league at home day 6, then a long break.
    cal = club_calendar(
        _games(
            [
                ("g1", 0, "league", 1, 2),
                ("g2", 3, "europe", 3, 1),
                ("g3", 6, "league", 1, 4),
                ("g4", 100, "league", 5, 1),
            ]
        )
    )
    club = cal.filter(pl.col("club_id") == 1).sort("date")
    assert club["rest_days_all"].to_list() == [MAX_GAP_DAYS, 3, 3, MAX_GAP_DAYS]
    assert club["games_7d_all"].to_list() == [0, 1, 2, 0]
    assert club["non_league_14d"].to_list() == [0, 0, 1, 0]
    assert club["prev_europe_away"].to_list() == [False, False, True, False]
    assert club["next_is_europe"].to_list() == [True, False, False, False]
    assert club["days_to_next_all"].to_list() == [3, 3, MAX_GAP_DAYS, None]
    assert club["is_home"].to_list() == [True, False, True, False]
