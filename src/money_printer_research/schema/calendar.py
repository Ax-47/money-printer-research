"""Full-calendar schedule frames: club_calendar's and team_schedule_features' outputs."""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

COMP_TYPES = ["league", "domestic_cup", "europe", "other"]


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
        nullable=True,
        ge=0,
        le=30,
        description=(
            "Days since the club's previous game, any competition, capped at 30 (MAX_GAP_DAYS):"
            " the raw gap reaches years for a club that spent seasons outside the five leagues."
            " Null for the club's first game."
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


class TeamScheduleCols(pa.DataFrameModel):
    """team_schedule_features: per Understat game_id, each side's ClubCalendarCols features.

    Null on a side whose club has no matched Transfermarkt game.
    """

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    h_rest_days_all: F[int] = pa.Field(nullable=True, description="Home rest_days_all.")
    h_games_7d_all: F[int] = pa.Field(nullable=True, description="Home games_7d_all.")
    h_games_14d_all: F[int] = pa.Field(nullable=True, description="Home games_14d_all.")
    h_non_league_14d: F[int] = pa.Field(nullable=True, description="Home non_league_14d.")
    h_prev_europe_away: F[bool] = pa.Field(nullable=True, description="Home prev_europe_away.")
    h_days_to_next_all: F[int] = pa.Field(nullable=True, description="Home days_to_next_all.")
    h_next_is_europe: F[bool] = pa.Field(nullable=True, description="Home next_is_europe.")
    a_rest_days_all: F[int] = pa.Field(nullable=True, description="Away rest_days_all.")
    a_games_7d_all: F[int] = pa.Field(nullable=True, description="Away games_7d_all.")
    a_games_14d_all: F[int] = pa.Field(nullable=True, description="Away games_14d_all.")
    a_non_league_14d: F[int] = pa.Field(nullable=True, description="Away non_league_14d.")
    a_prev_europe_away: F[bool] = pa.Field(nullable=True, description="Away prev_europe_away.")
    a_days_to_next_all: F[int] = pa.Field(nullable=True, description="Away days_to_next_all.")
    a_next_is_europe: F[bool] = pa.Field(nullable=True, description="Away next_is_europe.")


__all__ = ["COMP_TYPES", "ClubCalendarCols", "TeamScheduleCols"]
