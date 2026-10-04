import re

import pandas as pd
import polars as pl

_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def to_snake(name: str) -> str:
    """HomeTeam -> home_team, 'Home Team' -> home_team, 'Avg>2.5' -> avg_2_5."""
    name = _CAMEL_1.sub(r"\1_\2", name)
    name = _CAMEL_2.sub(r"\1_\2", name)
    name = _NON_ALNUM.sub("_", name)
    return name.strip("_").lower()


def _mapping(columns: list[str]) -> dict[str, str]:
    mapping = {c: to_snake(c) for c in columns}
    seen: dict[str, str] = {}
    for old, new in mapping.items():
        if not new:
            raise ValueError(f"Column {old!r} becomes empty after renaming")
        if new in seen:
            raise ValueError(f"Columns {seen[new]!r} and {old!r} both become {new!r}")
        seen[new] = old
    return mapping


def snake_columns_pl(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename(_mapping(df.columns))


def snake_columns_pd(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_mapping(list(df.columns)))
