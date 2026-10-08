"""Validation as Kleisli arrows: pandera schemas inside a Result pipeline.

    check(M)        K[pl.DataFrame, DataFrame[M]]: validate against model M, every
                    problem in one Err; use it as a step in compose(...) or >>
    @returns(M)     make a function that builds a pl.DataFrame return
                    Result[DataFrame[M]]: an exception or a schema failure is an Err

The only module of the package that imports pandera.
"""

from collections.abc import Callable
from functools import wraps

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors
from pandera.typing.polars import DataFrame

from .monoid import K
from .result import Err, Ok, Result, name_of


def check[M: pa.DataFrameModel](model: type[M]) -> K[pl.DataFrame, DataFrame[M]]:
    """Validate against `model` (every problem in one Err) and type the frame DataFrame[M]."""

    def run(df: pl.DataFrame) -> Result[DataFrame[M]]:
        try:
            return Ok(model.validate(df, lazy=True))
        except SchemaErrors as e:
            return Err(f"{model.__name__}:\n{e.failure_cases}", e)

    return K(run, f"check({model.__name__})")


def returns[M: pa.DataFrameModel, **P](
    model: type[M],
) -> Callable[[Callable[P, pl.DataFrame]], Callable[P, Result[DataFrame[M]]]]:
    """Check f's returned frame against `model`, giving Result[DataFrame[M]].

    Like a Rust fn returning Result<DataFrame<M>, Err>: an exception or a schema
    failure becomes Err naming f.
    """

    def decorate(f: Callable[P, pl.DataFrame]) -> Callable[P, Result[DataFrame[M]]]:
        validate = check(model)
        label = name_of(f)

        @wraps(f)
        def run(*args: P.args, **kwargs: P.kwargs) -> Result[DataFrame[M]]:
            try:
                df = f(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - every failure becomes an Err
                return Err(f"{label}: {e!r}", cause=e)
            return validate(df).map_err(lambda reason: f"{label} -> {reason}")

        return run

    return decorate


__all__ = ["check", "returns"]
