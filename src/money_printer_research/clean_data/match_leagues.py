"""Transfermarkt games + competitions -> match_leagues (MatchLeaguesCols)."""

import polars as pl
from pandera.typing.polars import DataFrame

from ..cool_stuff import returns
from ..schema import FORMATION_COLUMNS
from ..schema import GameFormationCols
from ..schema import MatchLeaguesCols
from ..schema import TransfermarktCompetitions
from ..schema import dtype

# UEFA club competitions, by id: Transfermarkt files EL and UCOL under type "other".
EUROPE = ("CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC")

LEAGUE = {
    "GB1": "ENG-Premier League",
    "ES1": "ESP-La Liga",
    "L1": "GER-Bundesliga",
    "IT1": "ITA-Serie A",
    "FR1": "FRA-Ligue 1",
}
# Domestic cups that competitions.parquet does not list (CGB: EFL Cup).
EXTRA_DOMESTIC_CUP = ("CGB",)


@returns(MatchLeaguesCols)
def join_match_w_league(
    matches: DataFrame[GameFormationCols], leagues: DataFrame[TransfermarktCompetitions]
) -> pl.DataFrame:
    """Transfermarkt games -> MatchLeaguesCols, with comp_type, Understat league and formations.

    comp_type: league / domestic_cup / europe / other.
    """
    cid = pl.col(GameFormationCols.competition_id)
    comp = leagues.select(
        TransfermarktCompetitions.competition_id,
        pl.col(TransfermarktCompetitions.type).alias("tm_type"),
    )
    comp_type = (
        pl.when(cid.is_in(list(LEAGUE)))
        .then(pl.lit("league"))
        .when(cid.is_in(EUROPE))
        .then(pl.lit("europe"))
        .when((pl.col("tm_type") == "domestic_cup") | cid.is_in(EXTRA_DOMESTIC_CUP))
        .then(pl.lit("domestic_cup"))
        .otherwise(pl.lit("other"))
    )
    return (
        matches.join(comp, on=GameFormationCols.competition_id, how="left")
        .select(
            pl.col(GameFormationCols.game_id).alias(MatchLeaguesCols.tm_game_id),
            pl.col(GameFormationCols.competition_id).alias(MatchLeaguesCols.competition_id),
            comp_type.alias(MatchLeaguesCols.comp_type),
            cid.replace_strict(LEAGUE, default=None).alias(MatchLeaguesCols.league),
            pl.col(GameFormationCols.date)
            .cast(dtype(MatchLeaguesCols, MatchLeaguesCols.date))
            .alias(MatchLeaguesCols.date),
            pl.col(GameFormationCols.home_club_id).alias(MatchLeaguesCols.home_club_id),
            pl.col(GameFormationCols.away_club_id).alias(MatchLeaguesCols.away_club_id),
            pl.col(GameFormationCols.home_club_goals)
            .cast(dtype(MatchLeaguesCols, MatchLeaguesCols.home_goals))
            .alias(MatchLeaguesCols.home_goals),
            pl.col(GameFormationCols.away_club_goals)
            .cast(dtype(MatchLeaguesCols, MatchLeaguesCols.away_goals))
            .alias(MatchLeaguesCols.away_goals),
            pl.col(GameFormationCols.season).alias(MatchLeaguesCols.season),
            pl.col(GameFormationCols.season)
            .cast(dtype(MatchLeaguesCols, MatchLeaguesCols.season_start))
            .alias(MatchLeaguesCols.season_start),
            pl.col(GameFormationCols.round).alias(MatchLeaguesCols.round),
            pl.col(GameFormationCols.home_club_formation).alias(MatchLeaguesCols.home_formation),
            pl.col(GameFormationCols.away_club_formation).alias(MatchLeaguesCols.away_formation),
            pl.col(*FORMATION_COLUMNS),  # same names in GF and ML, both from FormationCols
        )
        .filter(pl.col(MatchLeaguesCols.date).is_not_null())
    )
