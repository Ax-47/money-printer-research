"""Team names -> one canonical lowercase name per club, shared by every source.

A raw name is cleaned (lowercase, no dots, trimmed) and looked up in ALIASES.
ALIASES is also the allowlist: a cleaned name that is not a key is unknown, so
rename_team_names raises on it (an Err under auto lift) instead of letting a new
spelling silently split one club in two.

When a new season brings new names:
    print(alias_block(schedule))      # paste-ready lines, grouped by league
and paste them into _TABLE below, pointing any alias at its canonical name.
"""

import polars as pl

from money_printer_research.cool_stuff import Err, Ok, Result
from money_printer_research.schema import CLEANED_MATCH_STAT as S

from .aliases_team import ALIASES, TEAM_COLUMNS


def _clean(col: pl.Expr) -> pl.Expr:
    """Alias key of a raw name: lowercase, dots removed, trimmed."""
    return col.str.to_lowercase().str.replace_all(".", "", literal=True).str.strip_chars()


def _team_cols(df: pl.DataFrame) -> list[str]:
    """Team columns present in df; raises when there are none."""
    cols = [c for c in TEAM_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(f"no team column ({', '.join(TEAM_COLUMNS)}) in {df.columns}")
    return cols


def _team_keys(df: pl.DataFrame, *by: str) -> pl.DataFrame:
    """Distinct (*by, name, key) over every team column, key = _clean(name)."""
    return (
        pl.concat([df.select(*by, pl.col(c).alias("name")) for c in _team_cols(df)])
        .drop_nulls("name")
        .unique()
        .with_columns(_clean(pl.col("name")).alias("key"))
    )


def find_missing_aliases(df: pl.DataFrame) -> pl.DataFrame:
    """Team names whose cleaned key is not in ALIASES, as (name, key) sorted by name."""
    return _team_keys(df).filter(~pl.col("key").is_in(list(ALIASES))).sort("name")


def format_alias_lines(missing: dict[str, pl.DataFrame]) -> str:
    """Paste-ready ALIASES lines from {source: find_missing_aliases(...)}.

    The target defaults to the key itself; the comment names the sources it came from.
    """
    sources_by_key: dict[str, set[str]] = {}
    for src, df in missing.items():
        for key in df["key"].to_list():
            sources_by_key.setdefault(key, set()).add(src)
    return "\n".join(
        f'    "{key}": "{key}",  # {", ".join(sorted(srcs))}'
        for key, srcs in sorted(sources_by_key.items())
    )


def check_aliases(df: pl.DataFrame, source: str = "xg") -> Result[pl.DataFrame]:
    """Ok(df) when every team name is known, else Err with lines to paste into ALIASES."""
    missing = find_missing_aliases(df)
    if missing.is_empty():
        return Ok(df)
    return Err(
        f"{missing.height} team names missing from ALIASES:\n"
        + format_alias_lines({source: missing})
    )


def norm_team_names(df: pl.DataFrame) -> pl.DataFrame:
    """Replace every team column with its canonical name; raises on an unknown name."""
    return df.with_columns(_clean(pl.col(_team_cols(df))).replace_strict(ALIASES))


def alias_block(schedule: pl.DataFrame) -> str:
    """Paste-ready ALIASES lines for every schedule team not yet known, by league."""
    missing = (
        _team_keys(schedule, S.league)
        .filter(~pl.col("key").is_in(list(ALIASES)))
        .select(S.league, "key")
        .unique()
        .sort(S.league, "key")
    )
    lines: list[str] = []
    for (league,), group in missing.group_by(S.league, maintain_order=True):
        lines.append(f"    # {league}")
        lines += [f'    "{k}": "{k}",' for k in group["key"].to_list()]
    return "\n".join(lines)


__all__ = [
    "alias_block",
    "check_aliases",
    "find_missing_aliases",
    "format_alias_lines",
    "norm_team_names",
]
