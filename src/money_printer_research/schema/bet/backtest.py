"""What the backtest command writes and shows.

BacktestCols         backtest.csv: every game with odds, the model's H/D/A, its pick
ThresholdSeasonCols  walk_forward_thresholds: each season at a hindsight-free threshold
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData
from pandera.typing import FieldType as F

from money_printer_research.schema.match.features import OUTCOMES


class BacktestCols(pa.DataFrameModel):
    """backtest.csv: one row per game from backtest_from that has odds. Key: game_id."""

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    date: F[datetime.datetime] = pa.Field(description="Kick-off, as in cleaned_matches.")
    league: F[str] = pa.Field(description='Understat league: "GER-Bundesliga".')
    season_start: F[pl.Int32] = pa.Field(description="Year the season starts: 2024 for 2024/25.")
    home_team: F[str] = pa.Field(description="Home side, Understat name after norm_team_names.")
    away_team: F[str] = pa.Field(description="Away side, Understat name after norm_team_names.")
    full_time_result: F[str] = pa.Field(
        isin=list(OUTCOMES), description="H home win, D draw, A away win."
    )

    odds_H: F[float] = pa.Field(
        gt=1, description="Decimal odds on a home win; see ODDS_PREFERENCE."
    )
    odds_D: F[float] = pa.Field(gt=1, description="Decimal odds on a draw.")
    odds_A: F[float] = pa.Field(gt=1, description="Decimal odds on an away win.")

    p_H: F[float] = pa.Field(
        ge=0, le=1, description="Model's P(home win), fitted on earlier seasons only."
    )
    p_D: F[float] = pa.Field(ge=0, le=1, description="Model's P(draw).")
    p_A: F[float] = pa.Field(ge=0, le=1, description="Model's P(away win).")

    pick: F[str] = pa.Field(
        isin=list(OUTCOMES), description="The outcome with the highest expected value."
    )
    ev: F[float] = pa.Field(
        ge=-1, description="Expected profit of 1 unit on pick: p * odds - 1. 0.1 is +10%."
    )
    profit: F[float] = pa.Field(
        ge=-1, description="What 1 unit on pick returned: odds - 1 if it won, else -1."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def probabilities_sum_to_one(self, data: PolarsData) -> pl.LazyFrame:
        """p_H + p_D + p_A is 1 for every game."""
        total = pl.col("p_H") + pl.col("p_D") + pl.col("p_A")
        return data.lazyframe.select(((total - 1).abs() < 1e-6).alias("p_sum"))


class ThresholdSeasonCols(pa.DataFrameModel):
    """One row per season, bet at the threshold that made the most on the seasons before it.

    The first season has nothing before it, so it has no row. Key: season_start.
    """

    season_start: F[pl.Int32] = pa.Field(
        unique=True, description="Year the season starts: 2024 for 2024/25."
    )
    min_ev: F[float] = pa.Field(
        description="The threshold chosen: bet every pick with ev above it."
    )
    bets: F[int] = pa.Field(ge=0, description="Picks that cleared min_ev this season.")
    profit: F[float] = pa.Field(description="Units won or lost on them, 1 unit per bet.")
    roi: F[float] = pa.Field(
        nullable=True, description="profit / bets; null when nothing cleared min_ev."
    )
