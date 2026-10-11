import polars as pl
from pandera.typing.polars import DataFrame

from ..clean_data.normalize import add_season_start
from ..clean_data.rename_team_names import check_aliases, norm_team_names
from ..cool_stuff import K, check, compose, lift, returns
from ..schema import PlayerStatCols, UnderstatPlayerRaw


@returns(UnderstatPlayerRaw)
def _add_season_start(stat_df: DataFrame[UnderstatPlayerRaw]) -> pl.DataFrame:
    """Add season_start from the season label: "1415" -> 2014."""
    return add_season_start(
        stat_df, 2000 + pl.col(UnderstatPlayerRaw.season).str.slice(0, 2).cast(pl.Int32)
    )


PLAYER_STAT_STEPS = (
    _add_season_start,
    check_aliases,
    lift(norm_team_names),
    check(PlayerStatCols),
)
"""The morphisms of clean_player_stat, in order; trace(*PLAYER_STAT_STEPS) logs each one."""

clean_player_stat: K[DataFrame[UnderstatPlayerRaw], DataFrame[PlayerStatCols]] = compose(
    *PLAYER_STAT_STEPS
)
"""Raw Understat player stats -> with season_start and canonical team names."""
