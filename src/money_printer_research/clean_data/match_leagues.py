"""Transfermarkt games + competitions -> std_tm (MatchLeaguesCols)."""

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import FORMATION_COLUMNS
from money_printer_research.schema import GAME_FORMATION as GF
from money_printer_research.schema import MATCH_LEAGUE as ML
from money_printer_research.schema import TransfermarktCompetitions as C
from money_printer_research.schema import dtype

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


@returns(ML)
def join_match_w_league(matches: DataFrame[GF], leagues: DataFrame[C]) -> pl.DataFrame:
    """Transfermarkt games -> MatchLeaguesCols, with comp_type, Understat league and formations.

    comp_type: league / domestic_cup / europe / other.
    """
    cid = pl.col(GF.competition_id)
    comp = leagues.select(C.competition_id, pl.col(C.type).alias("tm_type"))
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
        matches.join(comp, on=GF.competition_id, how="left")
        .select(
            pl.col(GF.game_id).alias(ML.tm_game_id),
            pl.col(GF.competition_id).alias(ML.competition_id),
            comp_type.alias(ML.comp_type),
            cid.replace_strict(LEAGUE, default=None).alias(ML.league),
            pl.col(GF.date).cast(dtype(ML, ML.date)).alias(ML.date),
            pl.col(GF.home_club_id).alias(ML.home_club_id),
            pl.col(GF.away_club_id).alias(ML.away_club_id),
            pl.col(GF.home_club_goals).cast(dtype(ML, ML.home_goals)).alias(ML.home_goals),
            pl.col(GF.away_club_goals).cast(dtype(ML, ML.away_goals)).alias(ML.away_goals),
            pl.col(GF.season).alias(ML.season),
            pl.col(GF.season).cast(dtype(ML, ML.season_start)).alias(ML.season_start),
            pl.col(GF.round).alias(ML.round),
            pl.col(GF.home_club_formation).alias(ML.home_formation),
            pl.col(GF.away_club_formation).alias(ML.away_formation),
            pl.col(*FORMATION_COLUMNS),  # same names in GF and ML, both from FormationCols
        )
        .filter(pl.col(ML.date).is_not_null())
    )
