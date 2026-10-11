"""Check that a pipeline step did not damage the data it passed on.

compare(before, after, key) lines up a step's input and output on a key and counts what
a step should never do by accident: lose or add keys, duplicate a key, change a value
it was meant to carry over, or turn a value into null. changed_rows shows the rows
behind a count. Columns a step renames must be renamed in `before` first; columns it
rewrites on purpose (canonical team names) are left out of `cols`.

at_morphism(f) does the same from inside a pipeline: it is f as a Kleisli arrow that
logs (morphism, a, a.shape) for what f returned, after (None, a, a.shape) for the
pipeline's input (a Writer, cool_stuff.writer). Logged steps compose with >> like any K,
and log_report turns the log into a table:

    pipeline = at_morphism(f, key="game_id") >> at_morphism(g, key="game_id")
    logged = pipeline(told(df)).expect("pipeline")
    log_report(logged)   # rows: None (the input), f, g; shapes, nulls, keys lost
    logged.log[1].a      # the frame f returned

notebooks/data_integrity.ipynb runs these over every step of clean_data.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..cool_stuff import K, Logged, Morphism, compose, name_of, writer


_AFTER = "__after"


def _shared(before: pl.DataFrame, after: pl.DataFrame, key: Sequence[str]) -> list[str]:
    """Columns in both frames, the key left out, in `before`'s order."""
    return [c for c in before.columns if c in after.columns and c not in key]


def _cols(
    before: pl.DataFrame, after: pl.DataFrame, key: Sequence[str], cols: Sequence[str] | None
) -> list[str]:
    """Cols to compare: the shared columns by default, and never the key itself."""
    return _shared(before, after, key) if cols is None else [c for c in cols if c not in key]


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

    cols defaults to every column the two share; the key is left out either way. Every count but
    the heights should be 0 unless the step means it, such as a filter losing keys.
    """
    key = [key] if isinstance(key, str) else list(key)
    cols = _cols(before, after, key, cols)
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
    cols = _cols(before, after, key, cols)
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


@dataclass(frozen=True)
class Missing:
    """One log entry: log(morphism, a, a.shape), what a morphism returned and its shape.

    The first entry of a pipeline is its input, with morphism None; each later one is
    a morphism's output, so the entry before it is that morphism's input. shape is None
    when a is not a DataFrame. key, if given, is what log_report compares entries on.
    """

    morphism: str | None
    a: Any
    shape: tuple[int, int] | None
    key: tuple[str, ...] | None = None


def missing(
    morphism: str | None,
    a: Any,
    shape: tuple[int, int] | None,
    key: str | Sequence[str] | None = None,
) -> Missing:
    """The log entry log(morphism, a, a.shape); see Missing."""
    keys = (key,) if isinstance(key, str) else tuple(key) if key is not None else None
    return Missing(morphism, a, shape, keys)


def _entry(morphism: str | None, a: Any, key: str | Sequence[str] | None) -> Missing:
    """missing(morphism, a, a.shape), with no shape when a is not a frame."""
    return missing(morphism, a, a.shape if isinstance(a, pl.DataFrame) else None, key)


def at_morphism(
    f: Morphism[Any, Any], key: str | Sequence[str] | None = None
) -> K[Logged[Any, Missing], Logged[Any, Missing]]:
    """F as a logged arrow: run it, and log (f, a, a.shape) for what it returned.

    On an empty log it first logs the input as (None, a, a.shape), so

        (at_morphism(f) >> at_morphism(g))(told(df))

    logs (None, df, df.shape), (f, f(df), ...), (g, g(f(df)), ...).
    """
    name = name_of(f)

    def seed(x: Logged[Any, Missing]) -> Logged[Any, Missing]:
        return x if x.log else x.tell(_entry(None, x.value, key))

    logged = K(seed) >> writer(f, lambda _, b: _entry(name, b, key))
    return K(logged.run, name)


def trace(
    *steps: Morphism[Any, Any], key: str | Sequence[str] | None = None
) -> K[Logged[Any, Missing], Logged[Any, Missing]]:
    """at_morphism on every step, composed with >>: a whole pipeline, logged step by step."""
    return compose(*(at_morphism(f, key) for f in steps))


def _row(previous: Missing | None, entry: Missing) -> dict[str, Any]:
    """One row of log_report: entry's shape and nulls, and how they moved from previous."""
    a = entry.a if isinstance(entry.a, pl.DataFrame) else None
    before = previous.a if previous is not None and isinstance(previous.a, pl.DataFrame) else None
    nulls_out = _nulls(a) if a is not None else {}
    nulls_in = _nulls(before) if before is not None else {}
    counts: dict[str, Any] = dict.fromkeys(
        ("lost_keys", "new_keys", "dup_keys", "changed", "new_nulls")
    )
    if entry.key is not None and before is not None and a is not None:
        report = compare(before, a, list(entry.key)).row(0, named=True)
        counts = {k: report[k] for k in counts}
    return {
        "morphism": entry.morphism,
        "shape": list(entry.shape) if entry.shape else None,
        "nulls": sum(nulls_out.values()) if a is not None else None,
        "more_nulls": (
            [c for c, n in nulls_out.items() if n > nulls_in.get(c, 0)]
            if before is not None
            else None
        ),
        **counts,
    }


def log_report(logged: Logged[Any, Missing]) -> pl.DataFrame:
    """The log as a table, one row per entry in order: the input first, then each morphism.

    more_nulls lists the columns with more nulls than in the entry before, a column
    the morphism added counting from 0; with a key, lost_keys to new_nulls are compare's
    counts against the entry before. The first row has none of these.
    """
    entries = list(logged.log)
    rows = [_row(prev, cur) for prev, cur in zip([None, *entries], entries, strict=False)]
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = [
    "Missing",
    "at_morphism",
    "changed_rows",
    "compare",
    "disagree",
    "log_report",
    "missing",
    "trace",
]
