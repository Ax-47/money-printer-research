"""Pre-match fatigue features per player, and per-team summaries for matches.

All load features use only matches before the current one (closed="left"),
so they are known at kick-off. Understat covers league matches only, so rest
days are overstated for clubs that also play cups or European games.
"""

import polars as pl

FATIGUE_COLS = ["rest_days", "n_7d", "min_7d", "min_28d", "streak_prev", "fatigue_score"]


def _prev_window(col: pl.Expr, window: str) -> pl.Expr:
    """Sum of `col` over the previous `window`, excluding the current match."""
    return (
        col.rolling_sum_by("match_date", window_size=window, closed="left")
        .over("player_id")
        .fill_null(0)
    )


def _team_game_index(pm: pl.DataFrame) -> pl.DataFrame:
    """Order of each game within a team's season: 0, 1, 2, ..."""
    return (
        pm.select("team_id", "season_start", "game_id", "match_date")
        .unique()
        .sort("team_id", "match_date")
        .with_columns(pl.int_range(pl.len()).over("team_id", "season_start").alias("tidx"))
        .select("team_id", "game_id", "tidx")
    )


def player_fatigue(player_matches: pl.DataFrame) -> pl.DataFrame:
    """player_matches plus pre-match fatigue columns (see FATIGUE_COLS)."""
    pm = (
        player_matches.join(
            _team_game_index(player_matches), on=["team_id", "game_id"], validate="m:1"
        )
        .sort("player_id", "match_date")
        .with_columns(
            pl.col("match_date").diff().dt.total_days().over("player_id").alias("rest_days"),
            _prev_window(pl.col("minutes"), "7d").alias("min_7d"),
            _prev_window(pl.col("minutes").is_not_null().cast(pl.Int64), "7d").alias("n_7d"),
            _prev_window(pl.col("minutes"), "28d").alias("min_28d"),
            # A new streak starts when the player missed the team's previous game.
            (pl.col("tidx").diff().over("player_id", "season_start") != 1)
            .fill_null(True)
            .alias("_brk"),
        )
        .with_columns(
            pl.col("_brk")
            .cast(pl.Int32)
            .cum_sum()
            .over("player_id", "season_start")
            .alias("_streak")
        )
        .with_columns(
            pl.int_range(pl.len())
            .over("player_id", "season_start", "_streak")
            .alias("streak_prev")
        )
    )

    # Hand-set weights, not fitted. Each term is scaled to 0-1.
    chronic_week = pl.when(pl.col("min_28d") > 0).then(pl.col("min_28d") / 4)
    acwr = (pl.col("min_7d") / chronic_week).clip(0, 3).fill_null(1)
    score = (
        0.35 * (pl.col("min_7d") / 180).clip(0, 1)
        + 0.25 * ((7 - pl.col("rest_days").fill_null(7)) / 5).clip(0, 1)
        + 0.20 * (pl.col("streak_prev") / 6).clip(0, 1)
        + 0.20 * (acwr - 1).clip(0, 1)
    ) * 100

    return pm.with_columns(score.round(1).alias("fatigue_score")).drop("tidx", "_brk", "_streak")


def team_fatigue(fatigue: pl.DataFrame) -> pl.DataFrame:
    """Mean starter fatigue per side, one row per game_id."""
    starters = fatigue.filter(pl.col("position") != "Sub")
    return (
        starters.group_by("game_id")
        .agg(
            pl.col("fatigue_score").filter(pl.col("is_home")).mean().alias("h_fatigue"),
            pl.col("fatigue_score").filter(~pl.col("is_home")).mean().alias("a_fatigue"),
            pl.col("min_7d").filter(pl.col("is_home")).sum().alias("h_min_7d"),
            pl.col("min_7d").filter(~pl.col("is_home")).sum().alias("a_min_7d"),
        )
        .with_columns((pl.col("h_fatigue") - pl.col("a_fatigue")).alias("fatigue_diff"))
    )
