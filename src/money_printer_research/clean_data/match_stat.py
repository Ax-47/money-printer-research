import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data.normalize import norm_season_start
from money_printer_research.cool_stuff import K, check
from money_printer_research.schema import CLEANED_MATCH_STAT
from money_printer_research.schema.raw import MATCH_STAT


def _filter_only_result(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    return stat_df.filter(pl.col(MATCH_STAT.is_result))


def _norm_season_start_mt(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    return norm_season_start(
        stat_df,
        (2000 + pl.col(MATCH_STAT.season) // 100)
        .cast(pl.Int32)
        .alias(CLEANED_MATCH_STAT.season_start),
    )


match_stat_monad = (
    K(_filter_only_result)
    >> check(MATCH_STAT)
    >> _norm_season_start_mt
    >> check(CLEANED_MATCH_STAT)
)
# K[DataFrame[MATCH_STAT], DataFrame[ScheduleCols]]
