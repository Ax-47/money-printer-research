"""Plan A: every club's full match calendar (league, cups, Europe) from Transfermarkt.

Understat only has league games, so a team that played in Europe on Wednesday
looks rested for Saturday. Transfermarkt games (match_leagues) have all competitions.
club_calendar computes pre-match schedule features on that full calendar, and
team_schedule_features attaches them to each Understat game as h_* / a_* columns,
through the tm_game_id that join_match_w_stat put on cleaned_matches.

All features use only games before the match, except the look-ahead ones
(days_to_next_all, next_is_europe), which use fixtures already scheduled.
"""

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import (
    CLEANED_MATCH,
    MATCH_LEAGUE,
    ClubCalendarCols,
    TeamScheduleCols,
)

L = MATCH_LEAGUE
CC = ClubCalendarCols
TS = TeamScheduleCols

MAX_GAP_DAYS = 30
"""Cap on rest_days_all and days_to_next_all: beyond a month every club is fully rested,
and the raw gaps reach years for clubs that spent seasons outside the five leagues."""

SCHEDULE_FEATURES = [
    CC.rest_days_all,
    CC.games_7d_all,
    CC.games_14d_all,
    CC.non_league_14d,
    CC.prev_europe_away,
    CC.days_to_next_all,
    CC.next_is_europe,
]


def _prior_count(days: int, cond: pl.Expr) -> pl.Expr:
    """Games matching `cond` in the club's previous `days` days, the current one excluded."""
    return (
        cond.cast(pl.Int64)
        .rolling_sum_by(CC.date, window_size=f"{days}d", closed="left")
        .over(CC.club_id)
        .fill_null(0)
    )


def club_calendar(tm: DataFrame[MATCH_LEAGUE]) -> DataFrame[ClubCalendarCols]:
    """One row per club per game with the schedule features, computed on all competitions."""
    sides = [(L.home_club_id, True), (L.away_club_id, False)]
    cal = pl.concat(
        [
            tm.select(
                pl.col(L.tm_game_id).alias(CC.tm_game_id),
                pl.col(L.date).alias(CC.date),
                pl.col(L.comp_type).alias(CC.comp_type),
                pl.col(own).alias(CC.club_id),
                pl.lit(is_home).alias(CC.is_home),
            )
            for own, is_home in sides
        ]
    ).sort(CC.club_id, CC.date, CC.tm_game_id)

    comp, date = pl.col(CC.comp_type), pl.col(CC.date)
    any_game = date.is_not_null()
    europe_away = (comp == "europe") & ~pl.col(CC.is_home)
    out = cal.with_columns(
        date.diff().dt.total_days().over(CC.club_id).clip(0, MAX_GAP_DAYS).alias(CC.rest_days_all),
        _prior_count(7, any_game).alias(CC.games_7d_all),
        _prior_count(14, any_game).alias(CC.games_14d_all),
        _prior_count(14, comp != "league").alias(CC.non_league_14d),
        europe_away.shift(1).over(CC.club_id).fill_null(False).alias(CC.prev_europe_away),
        (date.shift(-1).over(CC.club_id) - date)
        .dt.total_days()
        .clip(0, MAX_GAP_DAYS)
        .alias(CC.days_to_next_all),
        (comp.shift(-1).over(CC.club_id) == "europe").fill_null(False).alias(CC.next_is_europe),
    )
    return ClubCalendarCols.validate(out, lazy=True)


@returns(TeamScheduleCols)
def team_schedule_features(
    tm: DataFrame[MATCH_LEAGUE], cleaned_matches: DataFrame[CLEANED_MATCH]
) -> DataFrame[TeamScheduleCols]:
    """Per Understat game_id: h_* and a_* schedule features from the full calendar."""
    game_map = cleaned_matches.select(CLEANED_MATCH.game_id, CLEANED_MATCH.tm_game_id).drop_nulls()
    cal = club_calendar(tm).join(game_map, on=CC.tm_game_id, how="inner", validate="m:1")

    def side(is_home: bool, prefix: str) -> pl.DataFrame:
        return cal.filter(pl.col(CC.is_home) == is_home).select(
            CLEANED_MATCH.game_id, *(pl.col(c).alias(f"{prefix}{c}") for c in SCHEDULE_FEATURES)
        )

    out = side(True, "h_").join(
        side(False, "a_"), on=CLEANED_MATCH.game_id, how="full", coalesce=True, validate="1:1"
    )
    return TeamScheduleCols.validate(out, lazy=True)
