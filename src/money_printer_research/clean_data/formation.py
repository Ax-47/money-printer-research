"""Encode Transfermarkt formation strings ("4-2-3-1", "4-4-2 double 6") as numbers.

A formation is a shape (players per line from the back, joined by "-") and an
optional variant after a space. encode_formation adds, for each side, the columns
of GameFormationCols: shape, variant, defenders, midfielders, forwards, lines.

    games >> encode_formation          # Result[DataFrame[GameFormationCols]]

A formation that does not parse gets null in every encoded column, so the schema
check passes it; unknown_formations lists those for ALIASES-style review.
"""

from typing import NamedTuple

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import returns
from money_printer_research.schema.formation import GAME_FORMATION as GF
from money_printer_research.schema.formation import OUTFIELD
from money_printer_research.schema.selected import TransfermarktGames as G

_SHAPE = r"^(\d(?:-\d)+)"


class Side(NamedTuple):
    """Column names for one side: the raw formation and its encoded columns."""

    formation: str
    shape: str
    variant: str
    defenders: str
    midfielders: str
    forwards: str
    lines: str


HOME = Side(
    G.home_club_formation,
    GF.home_shape,
    GF.home_variant,
    GF.home_defenders,
    GF.home_midfielders,
    GF.home_forwards,
    GF.home_lines,
)
AWAY = Side(
    G.away_club_formation,
    GF.away_shape,
    GF.away_variant,
    GF.away_defenders,
    GF.away_midfielders,
    GF.away_forwards,
    GF.away_lines,
)


def _lines(raw: pl.Expr) -> pl.Expr:
    """Parse the shape into players per line: "4-2-3-1 flat" -> [4, 2, 3, 1]."""
    return raw.str.extract(_SHAPE).str.split("-").cast(pl.List(pl.Int8))


def formation_exprs(side: Side) -> list[pl.Expr]:
    """Return the expressions that encode side.formation into side's columns."""
    raw = pl.col(side.formation).str.strip_chars()
    shape = raw.str.extract(_SHAPE)
    lines = _lines(raw)
    defenders = lines.list.first()
    forwards = lines.list.last()
    variant = raw.str.replace(_SHAPE, "").str.strip_chars().str.to_lowercase()
    return [
        shape.alias(side.shape),
        pl.when(shape.is_not_null() & (variant != "")).then(variant).alias(side.variant),
        defenders.alias(side.defenders),
        (OUTFIELD - defenders - forwards).cast(pl.Int8).alias(side.midfielders),
        forwards.alias(side.forwards),
        lines.list.len().cast(pl.Int8).alias(side.lines),
    ]


@returns(GF)
def encode_formation(games: DataFrame[G]) -> pl.DataFrame:
    """Add both sides' encoded formation columns, checked against GameFormationCols."""
    return games.with_columns(*formation_exprs(HOME), *formation_exprs(AWAY))


def unknown_formations(games: DataFrame[G]) -> pl.DataFrame:
    """List formations that do not parse or whose lines do not sum to 10 outfield players."""
    raw = (
        pl.concat(
            [
                games.select(pl.col(side.formation).str.strip_chars().alias("formation"))
                for side in (HOME, AWAY)
            ]
        )
        .drop_nulls()
        .unique()
    )
    lines = _lines(pl.col("formation"))
    return raw.filter(lines.is_null() | (lines.list.sum() != OUTFIELD)).sort("formation")


__all__ = ["AWAY", "HOME", "Side", "encode_formation", "formation_exprs", "unknown_formations"]
