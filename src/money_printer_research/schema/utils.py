import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors


def validate_lazy(df: pl.DataFrame, schema: pa.DataFrameSchema, name: str) -> pl.DataFrame:
    """Validate with every problem in one ValueError, as callers catch ValueError."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as e:
        raise ValueError(f"{name}:\n{e.failure_cases}") from e
