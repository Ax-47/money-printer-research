"""Missing values in match_features: where they come from, and how the table marks them.

A feature is null for one of four reasons:

    no line-up        fatigue and the player slots, for a side Understat has no starters for
    no TM game        the schedule features, for a side whose club has no Transfermarkt game
    too few minutes   a starter's per-90 rates, below MIN_PREV_MIN previous minutes
    no next game      days_to_next_all, for a club's last game in the Transfermarkt data

(A club's first game has no previous one either; club_calendar gives it MAX_GAP_DAYS of
rest rather than null, since the cap already means "a month or more".)

The nulls are kept: filling them with a value computed from the table (a mean, a median)
would leak later seasons into earlier ones, and a constant would pass for a real reading.
Instead add_missing_flags puts the reason on the row, so a model can tell "missing" from
"low", and null_report counts what is left.
"""

import re

import polars as pl

from money_printer_research.schema import N_SLOTS, slot_column

SIDES = ("h", "a")

LINEUP_PROBE = "min_7d"
"""h_min_7d / a_min_7d: null exactly when team_fatigue found no starters for that side."""

SCHEDULE_PROBE = "games_7d_all"
"""h_games_7d_all / a_games_7d_all: never null once a side has a Transfermarkt game."""

RATED_PROBE = "form90"
"""A starter has per-90 rates when his form90 slot is not null."""


def has_lineup(side: str) -> str:
    """Flag column: Understat has this side's starters ("h" -> "h_has_lineup")."""
    return f"{side}_has_lineup"


def has_schedule(side: str) -> str:
    """Flag column: this side's club has a matched Transfermarkt game ("h" -> "h_has_schedule")."""
    return f"{side}_has_schedule"


def rated_starters(side: str) -> str:
    """Count column: starters with per-90 rates ("h" -> "h_rated_starters")."""
    return f"{side}_rated_starters"


FLAG_COLUMNS = [
    col for side in SIDES for col in (has_lineup(side), has_schedule(side), rated_starters(side))
]


def add_missing_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add has_lineup, has_schedule and rated_starters for both sides; never null.

    Reads the fatigue, schedule and player slot columns that join_features puts on
    match_features, and leaves them as they are.
    """
    flags: list[pl.Expr] = []
    for side in SIDES:
        slots = range(1, N_SLOTS + 1)
        rated = [pl.col(slot_column(side, s, RATED_PROBE)).is_not_null() for s in slots]
        flags += [
            pl.col(f"{side}_{LINEUP_PROBE}").is_not_null().alias(has_lineup(side)),
            pl.col(f"{side}_{SCHEDULE_PROBE}").is_not_null().alias(has_schedule(side)),
            pl.sum_horizontal(rated).cast(pl.Int64).alias(rated_starters(side)),
        ]
    return df.with_columns(flags)


def feature_group(col: str) -> str:
    """Collapse sides and starter slots: "h_player3_xg90" -> "player_xg90", "a_x" -> "x"."""
    if m := re.fullmatch(r"[ha]_player\d+_(.+)", col):
        return f"player_{m[1]}"
    return re.sub(r"^[ha]_", "", col)


def null_report(df: pl.DataFrame) -> pl.DataFrame:
    """Nulls per feature group (see feature_group), most first; groups without nulls left out.

    share is the nulls over every cell of the group: rows times its number of columns.
    """
    counts = df.null_count().row(0, named=True)
    return (
        pl.DataFrame(
            {"group": [feature_group(c) for c in counts], "nulls": list(counts.values())},
            schema={"group": pl.String, "nulls": pl.Int64},
        )
        .group_by("group", maintain_order=True)
        .agg(pl.len().alias("columns"), pl.col("nulls").sum())
        .filter(pl.col("nulls") > 0)
        .with_columns(
            (pl.col("nulls") / (pl.col("columns") * max(df.height, 1))).round(4).alias("share")
        )
        .sort("nulls", descending=True, maintain_order=True)
    )


__all__ = [
    "FLAG_COLUMNS",
    "add_missing_flags",
    "feature_group",
    "has_lineup",
    "has_schedule",
    "null_report",
    "rated_starters",
]
