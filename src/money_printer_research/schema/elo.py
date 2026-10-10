import datetime

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData
from pandera.typing import FieldType as F

OUTCOMES = ("H", "D", "A")


class EloInputCols(pa.DataFrameModel):
    """What get_elo reads: one row per played match, built by to_elo_input."""

    season: F[str] = pa.Field(
        description='Season label: "2014/15". A change of season triggers promotion and regress.'
    )
    match_date: F[datetime.date] = pa.Field(
        description="Match date; get_elo replays in this order."
    )
    home_team: F[str] = pa.Field(description="Home side, Understat name after norm_team_names.")
    away_team: F[str] = pa.Field(description="Away side, Understat name after norm_team_names.")
    full_time_home_goals: F[int] = pa.Field(ge=0, description="Home goals at full time.")
    full_time_away_goals: F[int] = pa.Field(ge=0, description="Away goals at full time.")
    full_time_result: F[str] = pa.Field(
        isin=list(OUTCOMES), description="H home win, D draw, A away win; the target of evaluate."
    )


class EloCols(EloInputCols):
    """get_elo's output: the input plus pre-match ratings."""

    elo_home: F[float] = pa.Field(description="Home side's Elo rating before kick-off.")
    elo_away: F[float] = pa.Field(description="Away side's Elo rating before kick-off.")
    elo_diff: F[float] = pa.Field(
        description="elo_home - elo_away, without home_adv; the model's only regressor."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def diff_is_home_minus_away(self, data: PolarsData) -> pl.LazyFrame:
        """elo_diff equals elo_home - elo_away."""
        return data.lazyframe.select(
            ((pl.col("elo_home") - pl.col("elo_away") - pl.col("elo_diff")).abs() < 1e-9).alias(
                "elo_diff"
            )
        )


class PredictionCols(EloCols):
    """walk_forward's output: the test seasons with expected goals and H/D/A probabilities."""

    lam_home: F[float] = pa.Field(gt=0, description="Expected home goals (Poisson rate).")
    lam_away: F[float] = pa.Field(gt=0, description="Expected away goals (Poisson rate).")
    p_home: F[float] = pa.Field(ge=0, le=1, description="P(home win) from the score grid.")
    p_draw: F[float] = pa.Field(ge=0, le=1, description="P(draw) from the score grid.")
    p_away: F[float] = pa.Field(ge=0, le=1, description="P(away win) from the score grid.")
    rho: F[float] = pa.Field(
        description="Dixon-Coles rho fitted on the training seasons; 0.0 without Dixon-Coles."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def probabilities_sum_to_one(self, data: PolarsData) -> pl.LazyFrame:
        """p_home + p_draw + p_away is 1 for every match."""
        total = pl.col("p_home") + pl.col("p_draw") + pl.col("p_away")
        return data.lazyframe.select(((total - 1).abs() < 1e-9).alias("p_sum"))


__all__ = ["EloInputCols", "EloCols", "PredictionCols", "OUTCOMES"]
