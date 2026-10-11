"""Check that a pipeline step did not damage the data it passed on.

compare(before, after, key) lines up a step's input and output on a key and counts what
a step should never do by accident: lose or add keys, duplicate a key, change a value
it was meant to carry over, or turn a value into null. changed_rows shows the rows
behind a count. Columns a step renames must be renamed in `before` first; columns it
rewrites on purpose (canonical team names) are left out of `cols`.

at_morphism(f) does the same from inside a pipeline: it is f as a Kleisli arrow that
writes, at that morphism, missing(f, before, before.shape, after, after.shape) to a log
(a Writer, cool_stuff.writer). Logged steps compose with >> like any K, and log_report
turns the log into a table:

    pipeline = at_morphism(f, key="game_id") >> at_morphism(g, key="game_id")
    logged = pipeline(told(df)).expect("pipeline")
    log_report(logged)    # one row per morphism: shapes, nulls, keys lost
    logged.log[0].after   # the frame f returned

notebooks/data_integrity.ipynb runs these over every step of clean_data.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..cool_stuff import K, Logged, Morphism, compose, name_of, writer


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


@dataclass(frozen=True)
class Missing:
    """What one morphism did to the data: missing(f, before, before.shape, after, after.shape).

    The log entry at_morphism writes. It keeps both frames, so any step's input and
    output can be looked at afterwards; row() works out the counts from them.
    before is None when the morphism's input is not a frame (and no source picks one),
    after when its output is not.
    """

    morphism: str
    before: pl.DataFrame | None
    before_shape: tuple[int, int] | None
    after: pl.DataFrame | None
    after_shape: tuple[int, int] | None
    key: tuple[str, ...] | None = None

    def row(self) -> dict[str, Any]:
        """One row of log_report: shapes, nulls and, with a key, compare's counts.

        more_nulls lists the output columns with more nulls than in the input, a
        column the morphism added counting from 0.
        """
        before, after = self.before, self.after
        nulls_in = _nulls(before) if before is not None else {}
        nulls_out = _nulls(after) if after is not None else {}
        counts: dict[str, Any] = dict.fromkeys(
            ("lost_keys", "new_keys", "dup_keys", "changed", "new_nulls")
        )
        if self.key is not None and before is not None and after is not None:
            report = compare(before, after, list(self.key)).row(0, named=True)
            counts = {k: report[k] for k in counts}
        return {
            "morphism": self.morphism,
            "before_shape": list(self.before_shape) if self.before_shape else None,
            "after_shape": list(self.after_shape) if self.after_shape else None,
            "nulls_in": sum(nulls_in.values()) if before is not None else None,
            "nulls_out": sum(nulls_out.values()) if after is not None else None,
            "more_nulls": [c for c, n in nulls_out.items() if n > nulls_in.get(c, 0)],
            **counts,
        }


def missing(
    morphism: str,
    before: pl.DataFrame | None,
    before_shape: tuple[int, int] | None,
    after: pl.DataFrame | None,
    after_shape: tuple[int, int] | None,
    key: str | Sequence[str] | None = None,
) -> Missing:
    """The log entry for one morphism; see Missing."""
    keys = (key,) if isinstance(key, str) else tuple(key) if key is not None else None
    return Missing(morphism, before, before_shape, after, after_shape, keys)


def missing_note(
    morphism: str,
    key: str | Sequence[str] | None = None,
    source: Callable[[Any], pl.DataFrame] | None = None,
) -> Callable[[Any, Any], Missing]:
    """The note at_morphism writes: missing(morphism, before, before.shape, after, after.shape).

    The input frame is `source(input)` when given (for a morphism taking several
    frames), else the input itself if it is a DataFrame.
    """

    def note(a: Any, b: Any) -> Missing:
        before = source(a) if source is not None else a
        before = before if isinstance(before, pl.DataFrame) else None
        after = b if isinstance(b, pl.DataFrame) else None
        return missing(
            morphism,
            before,
            before.shape if before is not None else None,
            after,
            after.shape if after is not None else None,
            key,
        )

    return note


def at_morphism(
    f: Morphism[Any, Any],
    key: str | Sequence[str] | None = None,
    source: Callable[[Any], pl.DataFrame] | None = None,
) -> K[Logged[Any, Missing], Logged[Any, Missing]]:
    """F as a logged arrow: run it, and write a Missing entry for this morphism.

    at_morphism(f) >> at_morphism(g) applied to told(df) logs
    missing(f, df, df.shape, f(df), f(df).shape) then the same for g.
    """
    return writer(f, missing_note(name_of(f), key, source))


def trace(
    *steps: Morphism[Any, Any], key: str | Sequence[str] | None = None
) -> K[Logged[Any, Missing], Logged[Any, Missing]]:
    """at_morphism on every step, composed with >>: a whole pipeline, logged step by step."""
    return compose(*(at_morphism(f, key) for f in steps))


def log_report(logged: Logged[Any, Missing]) -> pl.DataFrame:
    """The log as a table, one row per morphism (Missing.row), in the order they ran."""
    return pl.DataFrame([m.row() for m in logged.log], infer_schema_length=None)


__all__ = [
    "Missing",
    "at_morphism",
    "changed_rows",
    "compare",
    "disagree",
    "log_report",
    "missing",
    "missing_note",
    "trace",
]
