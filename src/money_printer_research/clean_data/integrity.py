"""Check that a pipeline step did not damage the data it passed on.

compare(before, after, key) lines up a step's input and output on a key and counts what
a step should never do by accident: lose or add keys, duplicate a key, change a value
it was meant to carry over, or turn a value into null. changed_rows shows the rows
behind a count. Columns a step renames must be renamed in `before` first; columns it
rewrites on purpose (canonical team names) are left out of `cols`.

at_morphism(f) does the same from inside a pipeline: it is f as a Kleisli arrow that
writes, at that morphism, what happened to the rows, the keys and the nulls (a Writer,
cool_stuff.writer). Logged steps compose with >> like any K, and log_report turns the
log into a table:

    pipeline = at_morphism(f, key="game_id") >> at_morphism(g, key="game_id")
    log_report(pipeline(told(df)).expect("pipeline"))

notebooks/data_integrity.ipynb runs these over every step of clean_data.
"""

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

from ..cool_stuff import K, Logged, Morphism, compose, name_of, writer

type Entry = dict[str, Any]
"""One log entry: what a step did, as a row of log_report."""

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


def _nulls(df: pl.DataFrame) -> dict[str, int]:
    """Null count per column."""
    return df.null_count().row(0, named=True)


def missing_note(
    step: str,
    key: str | Sequence[str] | None = None,
    source: Callable[[Any], pl.DataFrame] | None = None,
) -> Callable[[Any, Any], Entry]:
    """The note at_morphism writes: rows, nulls and, with a key, compare's counts.

    The input frame is `source(input)` when given (for a step taking several frames),
    else the input itself if it is a DataFrame; without one, the *_in fields and
    compare's counts are None. more_nulls lists the output columns with more nulls
    than in the input, a column the step added counting from 0.
    """

    def note(a: Any, b: Any) -> Entry:
        before = source(a) if source is not None else a
        before = before if isinstance(before, pl.DataFrame) else None
        after = b if isinstance(b, pl.DataFrame) else None
        nulls_in = _nulls(before) if before is not None else {}
        nulls_out = _nulls(after) if after is not None else {}
        counts: Entry = dict.fromkeys(
            ("lost_keys", "new_keys", "dup_keys", "changed", "new_nulls")
        )
        if key is not None and before is not None and after is not None:
            counts = compare(before, after, key).drop("step").row(0, named=True)
        return {
            "step": step,
            "rows_in": before.height if before is not None else None,
            "rows_out": after.height if after is not None else None,
            "nulls_in": sum(nulls_in.values()) if before is not None else None,
            "nulls_out": sum(nulls_out.values()) if after is not None else None,
            "more_nulls": [c for c, n in nulls_out.items() if n > nulls_in.get(c, 0)],
            **{k: v for k, v in counts.items() if k not in ("rows_before", "rows_after")},
        }

    return note


def at_morphism(
    f: Morphism[Any, Any],
    key: str | Sequence[str] | None = None,
    source: Callable[[Any], pl.DataFrame] | None = None,
) -> K[Logged[Any, Entry], Logged[Any, Entry]]:
    """F as a logged arrow: run it, and write missing_note's entry for this morphism."""
    return writer(f, missing_note(name_of(f), key, source))


def trace(
    *steps: Morphism[Any, Any], key: str | Sequence[str] | None = None
) -> K[Logged[Any, Entry], Logged[Any, Entry]]:
    """at_morphism on every step, composed with >>: a whole pipeline, logged step by step."""
    return compose(*(at_morphism(f, key) for f in steps))


def log_report(logged: Logged[Any, Entry]) -> pl.DataFrame:
    """The log as a table, one row per morphism, in the order they ran."""
    return pl.DataFrame(list(logged.log), infer_schema_length=None)


__all__ = [
    "Entry",
    "at_morphism",
    "changed_rows",
    "compare",
    "disagree",
    "log_report",
    "missing_note",
    "trace",
]
