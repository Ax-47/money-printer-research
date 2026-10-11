"""Missing values in match_features, as their own table: missing.csv, one row per null cell.

match_features keeps its nulls and gains no columns. missing_table lists every null cell
(game_id, column) with the side it belongs to and the reason it is null:

    no_lineup         fatigue and the player slots, for a side Understat has no starters for
    short_lineup      a player slot left empty because fewer than 11 starters are recorded
    too_few_minutes   a starter's per-90 rates, below MIN_PREV_MIN previous minutes
    no_schedule       the schedule features, for a side whose club has no Transfermarkt game
    no_next_game      days_to_next_all, for a club's last game in the Transfermarkt data
    no_tm_game        the Transfermarkt columns, for a match with no Transfermarkt game
    no_formation      a side's formation columns, when its Transfermarkt game has no line-up
    plain_shape       home_variant / away_variant, for a shape with no variant ("4-2-3-1")
    other             none of these, such as a formation that did not parse

(A club's first game has no previous one either; club_calendar gives it MAX_GAP_DAYS of
rest rather than null, since the cap already means "a month or more".)

The nulls are kept: filling them with a value computed from the table (a mean, a median)
would leak later seasons into earlier ones, and a constant would pass for a real reading.
reason_counts summarises missing.csv, and null_report counts nulls per feature group.
"""

import re

import polars as pl

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import (
    CleanedMatchCols,
    FormationCols,
    MissingCols,
    ScheduleCols,
    TeamFatigueCols,
    TeamScheduleCols,
)

SIDES = ("h", "a")

LINEUP_PROBE = "min_7d"
"""h_min_7d / a_min_7d: null exactly when team_fatigue found no starters for that side."""

SCHEDULE_PROBE = "games_7d_all"
"""h_games_7d_all / a_games_7d_all: never null once a side has a Transfermarkt game."""

EMPTY_SLOT_PROBE = "prev_min"
"""A slot's prev_min is 0, not null, for any starter; null means the slot is empty."""

PER90 = ("xg90", "xa90", "form90")
"""The slot features that are null below MIN_PREV_MIN previous minutes."""

FATIGUE_COLUMNS = [c for c in TeamFatigueCols.to_schema().columns if c != "game_id"]
SCHEDULE_COLUMNS = [c for c in TeamScheduleCols.to_schema().columns if c != "game_id"]
TM_COLUMNS = [
    c for c in CleanedMatchCols.to_schema().columns if c not in ScheduleCols.to_schema().columns
]
"""cleaned_matches' columns from the Transfermarkt game: tm_*, formations."""

FORMATION_COLUMNS = [
    *FormationCols.to_schema().columns,
    CleanedMatchCols.home_formation,
    CleanedMatchCols.away_formation,
]
"""Each side's formation and its encoding (shape, variant, line counts)."""

_SIDE_PREFIX = {"h": ("h_", "home_"), "a": ("a_", "away_")}

_SLOT = r"^[ha]_player(\d+)_(.+)$"


@returns(MissingCols)
def missing_table(match_features: pl.DataFrame) -> pl.DataFrame:
    """One row per null cell of match_features: game_id, column, side and reason (MissingCols)."""
    key = CleanedMatchCols.game_id
    cells = (
        match_features.select(
            key, *(pl.col(c).is_null().alias(c) for c in match_features.columns if c != key)
        )
        .unpivot(index=key, variable_name="column", value_name="is_null")
        .filter(pl.col("is_null"))
        .drop("is_null")
    )
    sources = match_features.select(
        key,
        *(pl.col(f"{s}_{LINEUP_PROBE}").is_null().alias(f"{s}_no_lineup") for s in SIDES),
        *(pl.col(f"{s}_{SCHEDULE_PROBE}").is_null().alias(f"{s}_no_schedule") for s in SIDES),
        pl.col(CleanedMatchCols.tm_game_id).is_null().alias("no_tm_game"),
        *(
            pl.col(f"{p}_formation").is_null().alias(f"{s}_no_formation")
            for s, p in (("h", "home"), ("a", "away"))
        ),
    )
    column = pl.col("column")
    side = pl.col("side")
    per_side = {
        name: pl.when(side == "h").then(pl.col(f"h_{name}")).otherwise(pl.col(f"a_{name}"))
        for name in ("no_lineup", "no_schedule", "no_formation")
    }
    is_slot = pl.col("slot").is_not_null()
    empty_slot = (pl.col("feature") == EMPTY_SLOT_PROBE).any().over(key, "side", "slot") & is_slot
    either_no_lineup = pl.col("h_no_lineup") | pl.col("a_no_lineup")
    reason = (
        pl.when(is_slot & per_side["no_lineup"])
        .then(pl.lit("no_lineup"))
        .when(empty_slot)
        .then(pl.lit("short_lineup"))
        .when(is_slot & pl.col("feature").is_in(PER90))
        .then(pl.lit("too_few_minutes"))
        .when(column.is_in(FATIGUE_COLUMNS) & side.is_not_null() & per_side["no_lineup"])
        .then(pl.lit("no_lineup"))
        .when((column == "fatigue_diff") & either_no_lineup)
        .then(pl.lit("no_lineup"))
        .when(column.is_in(SCHEDULE_COLUMNS) & per_side["no_schedule"])
        .then(pl.lit("no_schedule"))
        .when(column.is_in(SCHEDULE_COLUMNS) & column.str.ends_with("_days_to_next_all"))
        .then(pl.lit("no_next_game"))
        .when(column.is_in(TM_COLUMNS) & pl.col("no_tm_game"))
        .then(pl.lit("no_tm_game"))
        .when(column.is_in(FORMATION_COLUMNS) & side.is_not_null() & per_side["no_formation"])
        .then(pl.lit("no_formation"))
        .when(column.str.ends_with("_variant") & column.is_in(FORMATION_COLUMNS))
        .then(pl.lit("plain_shape"))
        .otherwise(pl.lit("other"))
    )
    return (
        cells.join(sources, on=key, how="left")
        .with_columns(
            pl.coalesce(
                pl.when(pl.any_horizontal(column.str.starts_with(p) for p in prefixes)).then(
                    pl.lit(s)
                )
                for s, prefixes in _SIDE_PREFIX.items()
            ).alias("side"),
            column.str.extract(_SLOT, 1).alias("slot"),
            column.str.extract(_SLOT, 2).alias("feature"),
        )
        .with_columns(reason.alias("reason"))
        .select(key, "column", "side", "reason")
        .sort(key, maintain_order=True)
    )


def reason_counts(missing: pl.DataFrame) -> pl.DataFrame:
    """Null cells per reason in a missing table, most first, with the matches they touch."""
    return (
        missing.group_by("reason")
        .agg(pl.len().alias("cells"), pl.col("game_id").n_unique().alias("matches"))
        .sort("cells", "reason", descending=[True, False])
    )


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


__all__ = ["feature_group", "missing_table", "null_report", "reason_counts"]
