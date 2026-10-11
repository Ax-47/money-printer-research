import polars as pl
from pandera.typing.polars import DataFrame

from ..clean_data.normalize import add_season_start
from ..clean_data.rename_team_names import check_aliases, norm_team_names
from ..cool_stuff import K, check, compose, lift, returns
from ..schema import ScheduleCols
from ..schema.match.raw import UnderstatScheduleRaw


@returns(UnderstatScheduleRaw)
def _keep_played_matches(stat_df: DataFrame[UnderstatScheduleRaw]) -> pl.DataFrame:
    """Played matches only: fixtures never played (is_result False) are dropped."""
    return stat_df.filter(pl.col(UnderstatScheduleRaw.is_result))


@returns(UnderstatScheduleRaw)
def _keep_unplayed_matches(stat_df: DataFrame[UnderstatScheduleRaw]) -> pl.DataFrame:
    """Fixtures not played yet (is_result False): what `predict` forecasts."""
    return stat_df.filter(~pl.col(UnderstatScheduleRaw.is_result))


@returns(ScheduleCols)
def _add_season_start(stat_df: DataFrame[UnderstatScheduleRaw]) -> pl.DataFrame:
    """Add season_start from Understat's season code: 1415 -> 2014."""
    return add_season_start(
        stat_df, (2000 + pl.col(UnderstatScheduleRaw.season) // 100).cast(pl.Int32)
    )


MATCH_STAT_STEPS = (
    _keep_played_matches,
    check(UnderstatScheduleRaw),
    _add_season_start,
    check_aliases,
    lift(norm_team_names),
    check(ScheduleCols),
)
"""The morphisms of clean_match_stat, in order; trace(*MATCH_STAT_STEPS) logs each one."""

clean_match_stat: K[DataFrame[UnderstatScheduleRaw], DataFrame[ScheduleCols]] = compose(
    *MATCH_STAT_STEPS
)
"""Raw Understat schedule -> played matches with season_start and canonical team names."""

clean_upcoming: K[DataFrame[UnderstatScheduleRaw], DataFrame[ScheduleCols]] = compose(
    _keep_unplayed_matches, *MATCH_STAT_STEPS[1:]
)
"""Raw Understat schedule -> unplayed fixtures, cleaned the same way as the played matches."""
