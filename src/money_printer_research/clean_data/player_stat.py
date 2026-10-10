import polars as pl
from pandera.typing.polars import DataFrame

from ..clean_data.normalize import add_season_start
from ..clean_data.rename_team_names import check_aliases, norm_team_names
from ..cool_stuff import K, check, returns
from ..schema import CLEANED_PLAYER_STAT, PLAYER_STAT


@returns(PLAYER_STAT)
def _add_season_start(stat_df: DataFrame[PLAYER_STAT]) -> pl.DataFrame:
    """Add season_start from the season label: "1415" -> 2014."""
    return add_season_start(
        stat_df, 2000 + pl.col(PLAYER_STAT.season).str.slice(0, 2).cast(pl.Int32)
    )


clean_player_stat: K[DataFrame[PLAYER_STAT], DataFrame[CLEANED_PLAYER_STAT]] = (
    K(_add_season_start) >> check_aliases >> norm_team_names >> check(CLEANED_PLAYER_STAT)
)
"""Raw Understat player stats -> with season_start and canonical team names."""
