"""Transfermarkt games with each side's starting formation encoded as numbers."""

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData
from pandera.typing import FieldType as F

from money_printer_research.schema.selected import TransfermarktGames

OUTFIELD = 10
"""Outfield players in a formation: every shape's lines sum to this."""

SHAPE_PATTERN = r"^\d(-\d)+$"
"""A shape is digits joined by "-": "4-2-3-1"."""


class FormationCols(pa.DataFrameModel):
    """Each side's starting formation as shape, variant and line counts.

    A mixin: GameFormationCols and MatchLeaguesCols inherit these columns and the
    ten-outfield-players check. Every column is null when the side has no line-up.
    """

    home_shape: F[str] = pa.Field(
        nullable=True,
        str_matches=SHAPE_PATTERN,
        description='Home formation without its variant: "4-3-3 Attacking" -> "4-3-3".',
    )

    home_variant: F[str] = pa.Field(
        nullable=True,
        description=(
            'Home formation variant, lowercase: "attacking", "defending", "double 6", '
            '"diamond", "flat". Null for a plain shape such as "4-2-3-1".'
        ),
    )

    home_defenders: F[pl.Int8] = pa.Field(
        nullable=True, ge=3, le=5, description="Players in the home side's back line."
    )

    home_midfielders: F[pl.Int8] = pa.Field(
        nullable=True,
        ge=1,
        le=7,
        description=(
            "Home players between the back and front lines: 10 - defenders - forwards. "
            "4-2-3-1 and 4-5-1 both give 5; home_lines tells them apart. 4-1-5 gives 1."
        ),
    )

    home_forwards: F[pl.Int8] = pa.Field(
        nullable=True, ge=1, le=5, description="Players in the home side's front line."
    )

    home_lines: F[pl.Int8] = pa.Field(
        nullable=True, ge=3, le=5, description='Lines in the home shape: "4-2-3-1" has 4.'
    )

    away_shape: F[str] = pa.Field(
        nullable=True, str_matches=SHAPE_PATTERN, description="Away shape; see home_shape."
    )

    away_variant: F[str] = pa.Field(nullable=True, description="Away variant; see home_variant.")

    away_defenders: F[pl.Int8] = pa.Field(
        nullable=True, ge=3, le=5, description="Players in the away side's back line."
    )

    away_midfielders: F[pl.Int8] = pa.Field(
        nullable=True, ge=1, le=7, description="Away midfielders; see home_midfielders."
    )

    away_forwards: F[pl.Int8] = pa.Field(
        nullable=True, ge=1, le=5, description="Players in the away side's front line."
    )

    away_lines: F[pl.Int8] = pa.Field(
        nullable=True, ge=3, le=5, description="Lines in the away shape; see home_lines."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def lines_hold_ten_outfield_players(self, data: PolarsData) -> pl.LazyFrame:
        """Each parsed shape's lines sum to 10; a null shape passes."""
        return data.lazyframe.select(
            (
                pl.col(side).is_null()
                | (pl.col(side).str.split("-").cast(pl.List(pl.Int8)).list.sum() == OUTFIELD)
            ).alias(side)
            for side in ("home_shape", "away_shape")
        )


class GameFormationCols(TransfermarktGames, FormationCols):
    """TransfermarktGames plus FormationCols: the output of encode_formation."""


FORMATION_COLUMNS: tuple[str, ...] = tuple(FormationCols.to_schema().columns)
"""The encoded column names, in schema order: carry them with pl.col(*FORMATION_COLUMNS)."""

GAME_FORMATION = GameFormationCols

__all__ = [
    "FORMATION_COLUMNS",
    "GAME_FORMATION",
    "OUTFIELD",
    "SHAPE_PATTERN",
    "FormationCols",
    "GameFormationCols",
]
