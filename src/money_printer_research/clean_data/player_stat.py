import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data.normalize import norm_season_start
from money_printer_research.clean_data.rename_team_names import check_aliases, norm_team_names
from money_printer_research.cool_stuff import K, check, returns
from money_printer_research.schema import CLEANED_PLAYER_STAT, PLAYER_STAT


@returns(PLAYER_STAT)
def _norm_season_start(stat_df: DataFrame[PLAYER_STAT]) -> pl.DataFrame:
    """Add season_start: 1415 -> 2014."""
    return norm_season_start(
        stat_df,
        (2000 + pl.col(PLAYER_STAT.season).str.slice(0, 2).cast(pl.Int32)).alias(
            CLEANED_PLAYER_STAT.season_start
        ),
    )


player_stat_monad: K[DataFrame[PLAYER_STAT], DataFrame[CLEANED_PLAYER_STAT]] = (
    K(_norm_season_start) >> check_aliases >> norm_team_names >> check(CLEANED_PLAYER_STAT)
)
