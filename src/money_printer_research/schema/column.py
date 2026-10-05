"""Shared Column type for raw_schema.py and schema.py.

A Column is a str (the column name) that also carries its expected dtype and
whether it may be missing or contain nulls, so it works anywhere a column name
does: pl.col(PM.minutes), df.select(M.game_id, M.elo_diff), and so on.
"""

from dataclasses import fields
from typing import Any

import polars as pl


class Column(str):
    """A column name that also knows its dtype, nullability and whether it is required."""

    dtype: pl.DataType
    nullable: bool
    required: bool

    def __new__(
        cls, name: str, dtype: pl.DataType, *, nullable: bool = False, required: bool = True
    ) -> "Column":
        """Create a column spec; nullable and required default to a strict column."""
        obj = super().__new__(cls, name)
        obj.dtype = dtype
        obj.nullable = nullable
        obj.required = required
        return obj


STR, INT, FLOAT, BOOL = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
DATE, DATETIME = pl.Date(), pl.Datetime("us")


def columns(schema: Any) -> list[Column]:
    """All Column fields of a schema dataclass instance, in field order."""
    return [getattr(schema, f.name) for f in fields(schema)]


def dtypes(*cols: Column) -> dict[str, pl.DataType]:
    """{name: dtype}, e.g. pl.read_csv(path, schema_overrides=dtypes(*columns(PM)))."""
    return {str(c): c.dtype for c in cols}


def check(df: pl.DataFrame, cols: list[Column], name: str = "frame") -> None:
    """Raise one error listing every missing column, wrong dtype and unexpected null."""
    problems = []
    for col in cols:
        if col not in df.columns:
            if col.required:
                problems.append(f"missing {col!s}")
            continue
        if df.schema[col] != col.dtype:
            problems.append(f"{col!s}: expected {col.dtype}, got {df.schema[col]}")
        if not col.nullable and (n := df[col].null_count()):
            problems.append(f"{col!s}: {n} nulls")
    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


def require(df: pl.DataFrame, *cols: str) -> None:
    """Raise one error listing every column in `cols` that `df` does not have."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def validate(df: pl.DataFrame, schema: Any) -> None:
    """check() against every field of a schema dataclass instance."""
    check(df, columns(schema), type(schema).__name__)
