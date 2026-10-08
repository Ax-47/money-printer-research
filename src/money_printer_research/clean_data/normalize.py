import polars as pl


def norm_season_start(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return df.with_columns(expr.alias("season_start"))
