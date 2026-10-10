"""Pre-match load per player per match: player_fatigue's input and output.

Every load column counts only matches before the current one, so all are known
at kick-off.
"""

import datetime

import pandera.polars as pa
from pandera.typing import FieldType as F

from money_printer_research.schema.player.clean import PlayerStatCols


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


__all__ = ["FatigueInputCols", "PlayerFatigueCols"]
