from typing import Callable, cast

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors
from pandera.typing.polars import DataFrame


def validate_lazy(df: pl.DataFrame, schema: pa.DataFrameSchema, name: str) -> pl.DataFrame:
    """Validate with every problem in one ValueError, as callers catch ValueError."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as e:
        raise ValueError(f"{name}:\n{e.failure_cases}") from e


def as_[T: pa.DataFrameModel](model: type[T]) -> Callable[[pl.DataFrame], DataFrame[T]]:
    """Step that checks a frame against `model` and types it as DataFrame[model]."""

    def check(df: pl.DataFrame) -> DataFrame[T]:
        return cast(DataFrame[T], validate_lazy(df, model.to_schema(), model.__name__))

    return check
