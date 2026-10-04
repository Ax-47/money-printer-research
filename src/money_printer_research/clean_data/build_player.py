import polars as pl

PLAYER_FEATURES = ["xg90", "xa90", "form90", "prev_min"]
N_SLOTS = 11


def _player_form(pm: pl.DataFrame, window: int, prior_minutes: float) -> pl.DataFrame:
    """Pre-match per-90 features for each player, from previous games only."""
    total_min = float(pm["minutes"].sum())
    xg_rate = float(pm["xg"].sum()) / total_min
    xa_rate = float(pm["xa"].sum()) / total_min

    def prev_sum(col: str) -> pl.Expr:
        # shift(1) excludes the current match.
        return (
            pl.col(col).rolling_sum(window, min_samples=1).shift(1).over("player_id").fill_null(0)
        )

    def shrunk_per90(prev: str, rate: float) -> pl.Expr:
        # Pull players with few minutes toward the league rate.
        return 90 * (pl.col(prev) + rate * prior_minutes) / (pl.col("prev_min") + prior_minutes)

    return (
        pm.sort("player_id", "match_date")
        .with_columns(
            prev_sum("xg").alias("prev_xg"),
            prev_sum("xa").alias("prev_xa"),
            prev_sum("minutes").alias("prev_min"),
        )
        .with_columns(
            shrunk_per90("prev_xg", xg_rate).alias("xg90"),
            shrunk_per90("prev_xa", xa_rate).alias("xa90"),
        )
        .with_columns((pl.col("xg90") + pl.col("xa90")).alias("form90"))
    )


def _player_slots(form: pl.DataFrame) -> pl.DataFrame:
    """One row per game: h_player1_xg90 ... a_player11_prev_min, best form first."""
    ranked = (
        form.filter(pl.col("position") != "Sub")
        .with_columns(
            pl.when(pl.col("is_home")).then(pl.lit("h")).otherwise(pl.lit("a")).alias("side"),
            pl.col("form90")
            .rank("ordinal", descending=True)
            .over("game_id", "is_home")
            .alias("slot"),
        )
        .filter(pl.col("slot") <= N_SLOTS)
    )

    wide = (
        ranked.unpivot(
            index=["game_id", "side", "slot"],
            on=PLAYER_FEATURES,
            variable_name="feature",
            value_name="value",
        )
        .with_columns(pl.format("{}_player{}_{}", "side", "slot", "feature").alias("column"))
        .pivot(on="column", index="game_id", values="value")
    )

    # pivot does not guarantee column order, so fix it explicitly.
    ordered = [
        f"{side}_player{slot}_{feat}"
        for side in ("h", "a")
        for slot in range(1, N_SLOTS + 1)
        for feat in PLAYER_FEATURES
    ]
    return wide.select("game_id", *[c for c in ordered if c in wide.columns])


def player_form(
    player_matches: pl.DataFrame, window: int = 10, prior_minutes: float = 450.0
) -> pl.DataFrame:
    """player_matches plus pre-match xg90, xa90, form90, prev_min per row."""
    return _player_form(player_matches, window, prior_minutes)


def build_player_feat(form: pl.DataFrame) -> pl.DataFrame:
    """Pre-match player features, one row per game_id, for joining onto matches."""
    return _player_slots(form)
