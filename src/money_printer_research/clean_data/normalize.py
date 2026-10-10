import polars as pl

SEASON_START = "season_start"


def add_season_start(df: pl.DataFrame, start_year: pl.Expr) -> pl.DataFrame:
    """Add the season's start year (e.g. 2014 for 2014-15) as the season_start column."""
    return df.with_columns(start_year.alias(SEASON_START))
