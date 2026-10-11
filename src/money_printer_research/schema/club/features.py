"""club_calendar: every club's full match calendar with pre-match schedule features."""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.competition.comp_type import COMP_TYPES


class ClubCalendarCols(pa.DataFrameModel):
    """club_calendar: one row per club per Transfermarkt game, any competition."""

    tm_game_id: F[str] = pa.Field(
        description="Transfermarkt game id (match_leagues.csv tm_game_id)."
    )
    date: F[datetime.date] = pa.Field(description="Match date.")
    comp_type: F[str] = pa.Field(isin=COMP_TYPES, description="See MatchLeaguesCols.comp_type.")
    club_id: F[pl.Int32] = pa.Field(description="Transfermarkt club id of this side.")
    is_home: F[bool] = pa.Field(description="True when the club is the home side.")
    rest_days_all: F[int] = pa.Field(
        ge=0,
        le=30,
        description=(
            "Days since the club's previous game, any competition, capped at 30 (MAX_GAP_DAYS):"
            " the raw gap reaches years for a club that spent seasons outside the five leagues."
            " 30 for the club's first game in the data, which has no previous one."
        ),
    )
    games_7d_all: F[int] = pa.Field(
        ge=0, description="Games in the previous 7 days, any competition."
    )
    games_14d_all: F[int] = pa.Field(
        ge=0, description="Games in the previous 14 days, any competition."
    )
    non_league_14d: F[int] = pa.Field(
        ge=0, description="Cup, European or other games in the previous 14 days."
    )
    prev_europe_away: F[bool] = pa.Field(
        description="The club's previous game was an away European tie (travel)."
    )
    days_to_next_all: F[int] = pa.Field(
        nullable=True,
        ge=0,
        le=30,
        description=(
            "Days until the club's next game, any competition, capped at 30 (MAX_GAP_DAYS)."
            " Null for its last game. Uses the"
            " date the next game was played, which for a postponed game was not known at kick-off."
        ),
    )
    next_is_europe: F[bool] = pa.Field(description="The club's next game is European (rotation).")


__all__ = ["ClubCalendarCols"]
