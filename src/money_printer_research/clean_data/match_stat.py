import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data.normalize import norm_season_start
from money_printer_research.clean_data.rename_team_names import check_aliases, norm_team_names
from money_printer_research.cool_stuff import K, check, returns
from money_printer_research.schema import CLEANED_MATCH_STAT
from money_printer_research.schema.raw import MATCH_STAT


@returns(MATCH_STAT)
def _filter_only_result(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    """Played matches only: fixtures never played (is_result False) are dropped."""
    return stat_df.filter(pl.col(MATCH_STAT.is_result))


@returns(CLEANED_MATCH_STAT)
def _norm_season_start_mt(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    """Add season_start: 1415 -> 2014."""
    return norm_season_start(
        stat_df,
        (2000 + pl.col(MATCH_STAT.season) // 100)
        .cast(pl.Int32)
        .alias(CLEANED_MATCH_STAT.season_start),
    )


match_stat_monad: K[DataFrame[MATCH_STAT], DataFrame[CLEANED_MATCH_STAT]] = (
    K(_filter_only_result)
    >> check(MATCH_STAT)
    >> _norm_season_start_mt
    >> check_aliases
    >> norm_team_names
    >> check(CLEANED_MATCH_STAT)
)
