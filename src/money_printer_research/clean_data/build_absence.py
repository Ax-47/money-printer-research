"""Who is missing from a team's usual starting XI, and how good are they.

For every team and player we track a pre-match "start probability": an EWMA of
whether the player started each of the team's previous matches (a match the
player sat out counts as 0). Players with a high start probability who are not
in today's XI are the absences; weighting them by their pre-match form90 gives
the attacking quality the team is missing.

Lineups are treated as known before kick-off, as for the existing XI features.
"""

import polars as pl

START_HALF_LIFE = 5  # team matches
TAIL = 10  # keep a player on the roster this many team matches after his last game
REGULAR = 0.6  # start probability that makes a player a regular


def _roster_grid(pm: pl.DataFrame) -> pl.DataFrame:
    """One row per team match and roster player, with start_prob and quality."""
    starts = pm.select(
        "game_id",
        "match_date",
        "team",
        "is_home",
        "player_id",
        (pl.col("position") != "Sub").alias("started"),
        "form90",
    )
    games = (
        starts.select("team", "game_id", "match_date", "is_home")
        .unique()
        .sort("team", "match_date", "game_id")
        .with_columns(pl.int_range(pl.len()).over("team").alias("tidx"))
    )
    starts = starts.join(games.select("team", "game_id", "tidx"), on=["team", "game_id"])
    span = starts.group_by("team", "player_id").agg(
        pl.col("tidx").min().alias("first"), pl.col("tidx").max().alias("last")
    )
    started = pl.col("started").fill_null(False)
    return (
        span.join(games.select("team", "tidx", "game_id", "is_home"), on="team")
        .filter(pl.col("tidx").is_between(pl.col("first"), pl.col("last") + TAIL))
        .join(
            starts.select("team", "player_id", "tidx", "started", "form90"),
            on=["team", "player_id", "tidx"],
            how="left",
        )
        .sort("team", "player_id", "tidx")
        .with_columns(
            # shift(1): only the team's earlier matches decide who is a regular.
            started.cast(pl.Float64)
            .shift(1)
            .ewm_mean(half_life=START_HALF_LIFE)
            .over("team", "player_id")
            .fill_null(0.0)
            .alias("start_prob"),
            # form90 is pre-match; carry the last known value through games he missed.
            pl.col("form90")
            .forward_fill()
            .over("team", "player_id")
            .fill_null(0.0)
            .alias("quality"),
            started.alias("started"),
        )
    )


def team_absence(pm: pl.DataFrame) -> pl.DataFrame:
    """One row per game: missing quality and missing regulars for each side."""
    weighted = pl.col("start_prob") * pl.col("quality")
    absent = ~pl.col("started")
    per_side = (
        _roster_grid(pm)
        .group_by("game_id", "is_home")
        .agg(
            weighted.filter(absent).sum().alias("missing_q"),
            ((pl.col("start_prob") >= REGULAR) & absent)
            .sum()
            .cast(pl.Int64)
            .alias("missing_regulars"),
        )
        .with_columns(
            pl.when(pl.col("is_home")).then(pl.lit("h")).otherwise(pl.lit("a")).alias("side")
        )
    )
    wide = per_side.pivot(on="side", index="game_id", values=["missing_q", "missing_regulars"])
    return wide.select(
        "game_id",
        pl.col("missing_q_h").alias("h_missing_q"),
        pl.col("missing_q_a").alias("a_missing_q"),
        pl.col("missing_regulars_h").alias("h_missing_regulars"),
        pl.col("missing_regulars_a").alias("a_missing_regulars"),
        (pl.col("missing_q_h") - pl.col("missing_q_a")).alias("missing_q_diff"),
    )
