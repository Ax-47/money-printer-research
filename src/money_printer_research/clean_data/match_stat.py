import polars as pl
from pandera.typing.polars import DataFrame

from ..clean_data.normalize import add_season_start
from ..clean_data.rename_team_names import check_aliases, norm_team_names
from ..cool_stuff import K, check, returns
from ..schema import CLEANED_MATCH_STAT
from ..schema.match.raw import MATCH_STAT


@returns(MATCH_STAT)
def _keep_played_matches(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    """Played matches only: fixtures never played (is_result False) are dropped."""
    return stat_df.filter(pl.col(MATCH_STAT.is_result))


@returns(CLEANED_MATCH_STAT)
def _add_season_start(stat_df: DataFrame[MATCH_STAT]) -> pl.DataFrame:
    """Add season_start from Understat's season code: 1415 -> 2014."""
    return add_season_start(stat_df, (2000 + pl.col(MATCH_STAT.season) // 100).cast(pl.Int32))


clean_match_stat: K[DataFrame[MATCH_STAT], DataFrame[CLEANED_MATCH_STAT]] = (
    K(_keep_played_matches)
    >> check(MATCH_STAT)
    >> _add_season_start
    >> check_aliases
    >> norm_team_names
    >> check(CLEANED_MATCH_STAT)
)
"""Raw Understat schedule -> played matches with season_start and canonical team names."""
