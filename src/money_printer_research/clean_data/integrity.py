"""Check that a pipeline step did not damage the data it passed on.

compare(before, after, key) lines up a step's input and output on a key and counts what
a step should never do by accident: lose or add keys, duplicate a key, change a value
it was meant to carry over, or turn a value into null. changed_rows shows the rows
behind a count. Columns a step renames must be renamed in `before` first; columns it
rewrites on purpose (canonical team names) are left out of `cols`.

notebooks/data_integrity.ipynb runs these over every step of clean_data.
"""

from collections.abc import Sequence

import polars as pl

_AFTER = "__after"


def _shared(before: pl.DataFrame, after: pl.DataFrame, key: Sequence[str]) -> list[str]:
    """Columns in both frames, the key left out, in `before`'s order."""
    return [c for c in before.columns if c in after.columns and c not in key]


def _aligned(
    before: pl.DataFrame, after: pl.DataFrame, key: Sequence[str], cols: Sequence[str]
) -> pl.DataFrame:
    """Rows whose key is on both sides, with each col of `after` as <col>__after.

    `after` is cast to `before`'s dtypes (non-strict), so a value that does not survive
    the cast, as after a CSV round trip, shows up as a new null.
    """
    right = after.select(
        *key, *(pl.col(c).cast(before.schema[c], strict=False).alias(c + _AFTER) for c in cols)
    )
    return before.select(*key, *cols).join(right, on=list(key), how="inner")


def compare(
    before: pl.DataFrame,
    after: pl.DataFrame,
    key: str | Sequence[str],
    cols: Sequence[str] | None = None,
    step: str = "",
) -> pl.DataFrame:
    """One row: what happened to the data between `before` and `after`.

    rows_before, rows_after  heights
    lost_keys, new_keys      keys on only one side
    dup_keys                 keys repeated in `after` (a join that fanned out)
    changed                  cells in `cols` that differ, null versus value excluded
    new_nulls                cells in `cols` null in `after` but not in `before`

    cols defaults to every column the two share, the key left out. Every count but
    the heights should be 0 unless the step means it, such as a filter losing keys.
    """
    key = [key] if isinstance(key, str) else list(key)
    cols = _shared(before, after, key) if cols is None else list(cols)
    both = _aligned(before, after, key, cols)
    keys_before, keys_after = before.select(key).unique(), after.select(key).unique()

    def total(exprs: list[pl.Expr]) -> int:
        return int(both.select(pl.sum_horizontal(exprs).sum()).item()) if exprs else 0

    a, b = pl.col, lambda c: pl.col(c + _AFTER)
    changed = [(a(c).is_not_null() & b(c).is_not_null() & (a(c) != b(c))) for c in cols]
    new_nulls = [(a(c).is_not_null() & b(c).is_null()) for c in cols]
    return pl.DataFrame(
        {
            "step": [step],
            "rows_before": [before.height],
            "rows_after": [after.height],
            "lost_keys": [keys_before.join(keys_after, on=key, how="anti").height],
            "new_keys": [keys_after.join(keys_before, on=key, how="anti").height],
            "dup_keys": [after.height - keys_after.height],
            "changed": [total(changed)],
            "new_nulls": [total(new_nulls)],
        }
    )


def changed_rows(
    before: pl.DataFrame,
    after: pl.DataFrame,
    key: str | Sequence[str],
    cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Long form of compare's changed and new_nulls: key, column, before, after.

    Values are shown as strings so columns of every dtype fit in one table.
    """
    key = [key] if isinstance(key, str) else list(key)
    cols = _shared(before, after, key) if cols is None else list(cols)
    both = _aligned(before, after, key, cols)
    parts = [
        both.filter(
            pl.col(c).is_not_null()
            & (pl.col(c + _AFTER).is_null() | (pl.col(c) != pl.col(c + _AFTER)))
        ).select(
            *key,
            pl.lit(c).alias("column"),
            pl.col(c).cast(pl.String).alias("before"),
            pl.col(c + _AFTER).cast(pl.String).alias("after"),
        )
        for c in cols
    ]
    text = {"column": pl.String, "before": pl.String, "after": pl.String}
    empty = pl.DataFrame(schema={**before.select(key).schema, **text})
    return pl.concat([empty, *parts]) if parts else empty


def disagree(df: pl.DataFrame, pairs: dict[str, str]) -> pl.DataFrame:
    """Rows where a column and its partner, both non-null, differ.

    For two sources that should agree once joined: {"home_goals": "tm_home_goals"}.
    """
    differs = [
        pl.col(a).is_not_null() & pl.col(b).is_not_null() & (pl.col(a) != pl.col(b))
        for a, b in pairs.items()
    ]
    return df.filter(pl.any_horizontal(differs)) if differs else df.clear()


__all__ = ["changed_rows", "compare", "disagree"]
