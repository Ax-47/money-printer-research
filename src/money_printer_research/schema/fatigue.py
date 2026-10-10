"""Pre-match fatigue frames: player_fatigue's input and output, team_fatigue's output.

    MatchSideCols       get_fatigue's view of a match: date and home side
    FatigueInputCols    PlayerStatCols plus match_date and is_home (player_fatigue reads it)
    PlayerFatigueCols   FatigueInputCols plus the load columns (player_fatigue's output)
    TeamFatigueCols     starters' fatigue per side, one row per game_id (team_fatigue's output)

Every load column counts only matches before the current one, so all are known
at kick-off.
"""

import datetime

import pandera.polars as pa
from pandera.typing import FieldType as F

from money_printer_research.schema.clean import PlayerStatCols


class MatchSideCols(pa.DataFrameModel):
    """get_fatigue's view of a match: when it was played and who was at home."""

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    match_date: F[datetime.date] = pa.Field(description="Date part of the kick-off.")
    home_team: F[str] = pa.Field(description="Home side, after norm_team_names.")


class FatigueInputCols(PlayerStatCols):
    """PlayerStatCols plus the match's date and side: what player_fatigue reads."""

    match_date: F[datetime.date] = pa.Field(description="Date part of the match's kick-off.")
    is_home: F[bool] = pa.Field(description="True when team is the match's home_team.")


class PlayerFatigueCols(FatigueInputCols):
    """player_fatigue's output: the input plus pre-match load; only earlier matches count."""

    rest_days: F[int] = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Days since the player's previous match. Null on a debut. Overstated for clubs "
            "that also play cups or European games, which Understat does not cover."
        ),
    )
    n_7d: F[int] = pa.Field(ge=0, description="Matches played in the 7 days before this match.")
    min_7d: F[int] = pa.Field(ge=0, description="Minutes played in the 7 days before this match.")
    min_28d: F[int] = pa.Field(
        ge=0, description="Minutes played in the 28 days before this match."
    )
    streak_prev: F[int] = pa.Field(
        ge=0, description="Consecutive team games the player played right before this one."
    )
    fatigue_score: F[float] = pa.Field(
        ge=0,
        le=100,
        description=(
            "0 to 100, weighting min_7d, rest_days, streak_prev and min_7d against min_28d;"
            " hand-set weights in player_fatigue."
        ),
    )


class TeamFatigueCols(pa.DataFrameModel):
    """team_fatigue's output: starters' fatigue per side, one row per game_id."""

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    h_fatigue: F[float] = pa.Field(
        nullable=True, description="Mean fatigue_score of the home starters."
    )
    a_fatigue: F[float] = pa.Field(
        nullable=True, description="Mean fatigue_score of the away starters."
    )
    h_min_7d: F[int] = pa.Field(
        nullable=True, description="Sum of the home starters' min_7d. Null without home starters."
    )
    a_min_7d: F[int] = pa.Field(
        nullable=True, description="Sum of the away starters' min_7d. Null without away starters."
    )
    fatigue_diff: F[float] = pa.Field(nullable=True, description="h_fatigue - a_fatigue.")


__all__ = ["FatigueInputCols", "MatchSideCols", "PlayerFatigueCols", "TeamFatigueCols"]
