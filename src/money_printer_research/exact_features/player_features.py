"""Pre-match player features, one column per side, starter slot and feature (slot_columns).

Each starter gets his recent form (xG, xA and minutes over his previous FORM_WINDOW
matches) and his fatigue_score (player_fatigue); the eleven starters of each side are
ordered by line-up position (GK first) and spread into h_player1_xg90 ... a_player11_*.
Only matches before the current one count, so every value is known at kick-off.
"""

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import fallible
from money_printer_research.schema import (
    CLEANED_MATCH,
    CLEANED_PLAYER_STAT,
    N_SLOTS,
    PLAYER_SLOT_FEATURES,
    PLAYER_SLOTS,
    FatigueInputCols,
    PlayerFatigueCols,
    slot_columns,
)

from .fatigue import player_fatigue, player_matches

FI = FatigueInputCols

FORM_WINDOW = 10
"""Previous matches (any team, any season) that prev_xg, prev_xa and prev_min sum over."""

MIN_PREV_MIN = 270
"""Below this many previous minutes (three full games) the per-90 rates are null, not noise."""

POSITION_ORDER = (
    "GK", "DR", "DC", "DL", "DMR", "DML", "DMC", "MR", "MC", "ML",
    "AMR", "AMC", "AML", "FWR", "FW", "FWL",
)  # fmt: skip
"""Understat line-up positions from the back, the order of the starter slots (Understat's own)."""


def _prev(col: str) -> pl.Expr:
    """Sum of `col` over the player's previous FORM_WINDOW matches, the current one excluded."""
    return (
        pl.col(col)
        .rolling_sum(FORM_WINDOW, min_samples=1)
        .shift(1)
        .over(FI.player_id)
        .fill_null(0)
    )


def player_form(fatigue: DataFrame[PlayerFatigueCols]) -> pl.DataFrame:
    """Add prev_xg, prev_xa, prev_min and the per-90 rates xg90, xa90 and form90.

    form90 is (prev_xg + prev_xa) per 90: the player's expected goal involvement.
    """
    per90 = pl.when(pl.col("prev_min") >= MIN_PREV_MIN).then(90 / pl.col("prev_min"))
    return (
        fatigue.sort(FI.player_id, FI.match_date, FI.game_id)
        .with_columns(
            _prev(FI.xg).alias("prev_xg"),
            _prev(FI.xa).alias("prev_xa"),
            _prev(FI.minutes).alias("prev_min"),
        )
        .with_columns(
            (pl.col("prev_xg") * per90).alias("xg90"),
            (pl.col("prev_xa") * per90).alias("xa90"),
            ((pl.col("prev_xg") + pl.col("prev_xa")) * per90).alias("form90"),
        )
    )


def player_slots(form: pl.DataFrame) -> pl.DataFrame:
    """One row per game_id: each side's starters, ordered GK first, as slot columns."""
    starters = (
        form.filter(pl.col(FI.position) != "Sub")
        .with_columns(
            pl.when(pl.col(FI.is_home)).then(pl.lit("h")).otherwise(pl.lit("a")).alias("side"),
            pl.col(FI.position)
            .replace_strict({p: i for i, p in enumerate(POSITION_ORDER)})
            .alias("_order"),
        )
        .sort(FI.game_id, "side", "_order", FI.player_id)
        .with_columns((pl.int_range(pl.len()).over(FI.game_id, "side") + 1).alias("slot"))
        .filter(pl.col("slot") <= N_SLOTS)
    )
    long = starters.unpivot(
        index=[FI.game_id, "side", "slot"], on=list(PLAYER_SLOT_FEATURES), variable_name="feature"
    ).with_columns(
        pl.format("{}_player{}_{}", "side", "slot", "feature").alias("column"),
        pl.col("value").cast(pl.Float64),
    )
    wide = long.pivot(on="column", index=FI.game_id, values="value")
    expected = list(slot_columns())
    missing = [pl.lit(None, pl.Float64).alias(c) for c in expected if c not in wide.columns]
    return wide.with_columns(missing).select(FI.game_id, *expected)


@fallible
def get_player_features(
    player_stat: DataFrame[CLEANED_PLAYER_STAT], matches: DataFrame[CLEANED_MATCH]
) -> pl.DataFrame:
    """Per Understat game_id: the slot columns for both sides' starters (PLAYER_SLOTS)."""
    pm = player_matches(player_stat, matches)
    out = player_slots(player_form(player_fatigue(pm)))
    return PLAYER_SLOTS.validate(out, lazy=True)


__all__ = [
    "FORM_WINDOW",
    "MIN_PREV_MIN",
    "POSITION_ORDER",
    "get_player_features",
    "player_form",
    "player_slots",
]
