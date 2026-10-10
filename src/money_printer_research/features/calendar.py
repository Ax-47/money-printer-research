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
    CleanedMatchCols,
    MatchLeaguesCols,
    ClubCalendarCols,
    TeamScheduleCols,
)


MAX_GAP_DAYS = 30
"""Cap on rest_days_all and days_to_next_all: beyond a month every club is fully rested,
and the raw gaps reach years for clubs that spent seasons outside the five leagues."""

SCHEDULE_FEATURES = [
    ClubCalendarCols.rest_days_all,
    ClubCalendarCols.games_7d_all,
    ClubCalendarCols.games_14d_all,
    ClubCalendarCols.non_league_14d,
    ClubCalendarCols.prev_europe_away,
    ClubCalendarCols.days_to_next_all,
    ClubCalendarCols.next_is_europe,
]


def _prior_count(days: int, cond: pl.Expr) -> pl.Expr:
    """Games matching `cond` in the club's previous `days` days, the current one excluded."""
    return (
        cond.cast(pl.Int64)
        .rolling_sum_by(ClubCalendarCols.date, window_size=f"{days}d", closed="left")
        .over(ClubCalendarCols.club_id)
        .fill_null(0)
    )


def club_calendar(tm: DataFrame[MatchLeaguesCols]) -> DataFrame[ClubCalendarCols]:
    """One row per club per game with the schedule features, computed on all competitions."""
    sides = [(MatchLeaguesCols.home_club_id, True), (MatchLeaguesCols.away_club_id, False)]
    cal = pl.concat(
        [
            tm.select(
                pl.col(MatchLeaguesCols.tm_game_id).alias(ClubCalendarCols.tm_game_id),
                pl.col(MatchLeaguesCols.date).alias(ClubCalendarCols.date),
                pl.col(MatchLeaguesCols.comp_type).alias(ClubCalendarCols.comp_type),
                pl.col(own).alias(ClubCalendarCols.club_id),
                pl.lit(is_home).alias(ClubCalendarCols.is_home),
            )
            for own, is_home in sides
        ]
    ).sort(ClubCalendarCols.club_id, ClubCalendarCols.date, ClubCalendarCols.tm_game_id)

    comp, date = pl.col(ClubCalendarCols.comp_type), pl.col(ClubCalendarCols.date)
    any_game = date.is_not_null()
    europe_away = (comp == "europe") & ~pl.col(ClubCalendarCols.is_home)
    out = cal.with_columns(
        date.diff()
        .dt.total_days()
        .over(ClubCalendarCols.club_id)
        .clip(0, MAX_GAP_DAYS)
        .alias(ClubCalendarCols.rest_days_all),
        _prior_count(7, any_game).alias(ClubCalendarCols.games_7d_all),
        _prior_count(14, any_game).alias(ClubCalendarCols.games_14d_all),
        _prior_count(14, comp != "league").alias(ClubCalendarCols.non_league_14d),
        europe_away.shift(1)
        .over(ClubCalendarCols.club_id)
        .fill_null(False)
        .alias(ClubCalendarCols.prev_europe_away),
        (date.shift(-1).over(ClubCalendarCols.club_id) - date)
        .dt.total_days()
        .clip(0, MAX_GAP_DAYS)
        .alias(ClubCalendarCols.days_to_next_all),
        (comp.shift(-1).over(ClubCalendarCols.club_id) == "europe")
        .fill_null(False)
        .alias(ClubCalendarCols.next_is_europe),
    )
    return ClubCalendarCols.validate(out, lazy=True)


@returns(TeamScheduleCols)
def team_schedule_features(
    tm: DataFrame[MatchLeaguesCols], cleaned_matches: DataFrame[CleanedMatchCols]
) -> DataFrame[TeamScheduleCols]:
    """Per Understat game_id: h_* and a_* schedule features from the full calendar."""
    game_map = cleaned_matches.select(
        CleanedMatchCols.game_id, CleanedMatchCols.tm_game_id
    ).drop_nulls()
    cal = club_calendar(tm).join(
        game_map, on=ClubCalendarCols.tm_game_id, how="inner", validate="m:1"
    )

    def side(is_home: bool, prefix: str) -> pl.DataFrame:
        return cal.filter(pl.col(ClubCalendarCols.is_home) == is_home).select(
            CleanedMatchCols.game_id, *(pl.col(c).alias(f"{prefix}{c}") for c in SCHEDULE_FEATURES)
        )

    out = side(True, "h_").join(
        side(False, "a_"), on=CleanedMatchCols.game_id, how="full", coalesce=True, validate="1:1"
    )
    return TeamScheduleCols.validate(out, lazy=True)
