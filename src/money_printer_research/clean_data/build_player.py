"""Pre-match player form, and the per-slot player columns of matches.csv."""

import polars as pl

from money_printer_research.schema import N_SLOTS, PLAYER_SLOT_FEATURES, PM, slot_columns


def player_form(pm: pl.DataFrame, window: int = 10, prior_minutes: float = 450.0) -> pl.DataFrame:
    """player_matches plus pre-match prev_xg, prev_xa, prev_min, xg90, xa90, form90.

    Rates come from the player's previous `window` matches only, shrunk toward
    the overall rate by `prior_minutes` of pseudo-data.
    """
    minutes = float(pm[PM.minutes].sum())
    xg_rate = float(pm[PM.xg].sum()) / minutes
    xa_rate = float(pm[PM.xa].sum()) / minutes

    def prev_sum(col: str) -> pl.Expr:
        # shift(1) excludes the current match.
        return (
            pl.col(col).rolling_sum(window, min_samples=1).shift(1).over(PM.player_id).fill_null(0)
        )

    def shrunk_per90(prev: str, rate: float) -> pl.Expr:
        # Pull players with few minutes toward the overall rate.
        return 90 * (pl.col(prev) + rate * prior_minutes) / (pl.col(PM.prev_min) + prior_minutes)

    return (
        pm.sort(PM.player_id, PM.match_date)
        .with_columns(
            prev_sum(PM.xg).alias(PM.prev_xg),
            prev_sum(PM.xa).alias(PM.prev_xa),
            prev_sum(PM.minutes).alias(PM.prev_min),
        )
        .with_columns(
            shrunk_per90(PM.prev_xg, xg_rate).alias(PM.xg90),
            shrunk_per90(PM.prev_xa, xa_rate).alias(PM.xa90),
        )
        .with_columns((pl.col(PM.xg90) + pl.col(PM.xa90)).alias(PM.form90))
    )


def _player_slots(form: pl.DataFrame) -> pl.DataFrame:
    """One row per game: h_player1_xg90 ... a_player11_prev_min, best form first."""
    ranked = (
        form.filter(pl.col(PM.position) != "Sub")
        .with_columns(
            pl.when(pl.col(PM.is_home)).then(pl.lit("h")).otherwise(pl.lit("a")).alias("side"),
            pl.col(PM.form90)
            .rank("ordinal", descending=True)
            .over(PM.game_id, PM.is_home)
            .alias("slot"),
        )
        .filter(pl.col("slot") <= N_SLOTS)
    )
    wide = (
        ranked.unpivot(
            index=[PM.game_id, "side", "slot"],
            on=list(PLAYER_SLOT_FEATURES),
            variable_name="feature",
            value_name="value",
        )
        # Same naming as schema.slot_column.
        .with_columns(pl.format("{}_player{}_{}", "side", "slot", "feature").alias("column"))
        .pivot(on="column", index=PM.game_id, values="value")
    )
    # pivot does not guarantee column order; slot_columns() is the canonical one.
    return wide.select(PM.game_id, *[c for c in slot_columns() if c in wide.columns])


def build_player_feat(form: pl.DataFrame) -> pl.DataFrame:
    """Pre-match player slot features, one row per game_id, for joining onto matches."""
    return _player_slots(form)
