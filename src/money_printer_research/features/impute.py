"""model_frame: match_features with the gaps that have a known meaning filled, ready to fit.

match_features keeps every null (see features.missing); this is the copy a model reads.
Each step answers one reason of missing.csv:

    no_lineup         drop the match: Understat has no starters for a side
    no_formation      the side's formation from its team's previous match, flagged
                      *_formation_imputed; a team with no earlier formation stays null
    no_next_game      days_to_next_all = MAX_GAP_DAYS: the club's last game before the summer
                      break, the same cap a club's first game gets for its rest
    plain_shape       variant "none": the shape has no variant, nothing is missing
    too_few_minutes   kept null, as is short_lineup: HistGradientBoosting takes NaN, and the
                      first season, where these pile up, is dropped as warm-up

The previous match only, never the team's most common formation: that would read later games.
"""

import polars as pl

from money_printer_research.schema import CleanedMatchCols, FormationCols

from .calendar import MAX_GAP_DAYS
from .missing import LINEUP_PROBE

SIDES = (("h", "home"), ("a", "away"))

WARMUP_SEASONS = 1
"""Seasons dropped from the start: every player and team starts without history in 2014/15."""

PLAIN_VARIANT = "none"

FORMATION_FIELDS = [
    "formation",
    *(c.removeprefix("home_") for c in FormationCols.to_schema().columns if c.startswith("home_")),
]
"""A side's formation columns without the home_ / away_ prefix: formation, shape, variant..."""


def drop_no_lineup(df: pl.DataFrame) -> pl.DataFrame:
    """Matches where either side has no Understat starters, removed."""
    return df.filter(
        pl.all_horizontal(pl.col(f"{s}_{LINEUP_PROBE}").is_not_null() for s, _ in SIDES)
    )


def fill_formation(df: pl.DataFrame) -> pl.DataFrame:
    """A side with no formation takes its team's formation from the team's previous match.

    Adds h_formation_imputed / a_formation_imputed. Matches are ordered by kick-off, so the
    filled value is always known before the match.
    """
    key, date = CleanedMatchCols.game_id, CleanedMatchCols.date
    known = pl.col("formation").is_not_null()
    latest = pl.when(known).then(pl.struct(FORMATION_FIELDS)).forward_fill().over("team")
    filled = (
        pl.concat(
            df.select(
                key,
                date,
                pl.lit(s).alias("side"),
                pl.col(f"{p}_team").alias("team"),
                *(pl.col(f"{p}_{f}").alias(f) for f in FORMATION_FIELDS),
            )
            for s, p in SIDES
        )
        .sort("team", date, key)
        .select(
            key, "side", (~known & latest.is_not_null()).alias("imputed"), latest.struct.unnest()
        )
    )
    out = df.drop(f"{p}_{f}" for _, p in SIDES for f in FORMATION_FIELDS)
    for s, p in SIDES:
        side = filled.filter(pl.col("side") == s).select(
            key,
            *(pl.col(f).alias(f"{p}_{f}") for f in FORMATION_FIELDS),
            pl.col("imputed").alias(f"{s}_formation_imputed"),
        )
        out = out.join(side, on=key, how="left")
    return out.select(*df.columns, *(f"{s}_formation_imputed" for s, _ in SIDES))


def fill_next_game(df: pl.DataFrame) -> pl.DataFrame:
    """days_to_next_all for a club's last game in the data: MAX_GAP_DAYS."""
    return df.with_columns(
        pl.col(f"{s}_days_to_next_all").fill_null(MAX_GAP_DAYS) for s, _ in SIDES
    )


def fill_plain_variant(df: pl.DataFrame) -> pl.DataFrame:
    """Variant "none" where the side has a shape but no variant ("4-2-3-1")."""
    return df.with_columns(
        pl.when(pl.col(f"{p}_shape").is_not_null() & pl.col(f"{p}_variant").is_null())
        .then(pl.lit(PLAIN_VARIANT))
        .otherwise(pl.col(f"{p}_variant"))
        .alias(f"{p}_variant")
        for _, p in SIDES
    )


def drop_warmup(df: pl.DataFrame, seasons: int = WARMUP_SEASONS) -> pl.DataFrame:
    """The first `seasons` seasons removed (season_start), after they have fed the history."""
    season = pl.col(CleanedMatchCols.season_start)
    return df.filter(season >= season.min() + seasons)


def model_frame(
    match_features: pl.DataFrame, warmup_seasons: int = WARMUP_SEASONS
) -> pl.DataFrame:
    """match_features with each known gap handled (see the module docstring), warm-up dropped.

    The formation fill runs before the warm-up season is dropped, so 2015/16 can borrow
    from 2014/15.
    """
    return (
        match_features.pipe(drop_no_lineup)
        .pipe(fill_formation)
        .pipe(fill_next_game)
        .pipe(fill_plain_variant)
        .pipe(drop_warmup, warmup_seasons)
    )


__all__ = [
    "drop_no_lineup",
    "drop_warmup",
    "fill_formation",
    "fill_next_game",
    "fill_plain_variant",
    "model_frame",
]
