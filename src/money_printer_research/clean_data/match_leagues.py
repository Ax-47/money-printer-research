from typing import cast

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import Morphism
from money_printer_research.schema import dtype
from money_printer_research.schema.clean import MATCH_LEAGUE as ML
from money_printer_research.schema.selected import TransfermarktCompetitions as C
from money_printer_research.schema.selected import TransfermarktGames as G

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


def _join_match_w_league(matches: DataFrame[G], leagues: DataFrame[C]) -> DataFrame[ML]:
    """Transfermarkt games -> schema.TM, with comp_type and Understat league.

    comp_type: league / domestic_cup / europe / other.
    """
    cid = pl.col(G.competition_id)

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
    out = (
        matches.join(comp, on=G.competition_id, how="left")
        .select(
            pl.col(G.game_id).alias(ML.tm_game_id),
            pl.col(G.competition_id).alias(ML.competition_id),
            comp_type.alias(ML.comp_type),
            pl.col(G.competition_id).replace_strict(LEAGUE, default=None).alias(ML.league),
            pl.col(G.date).cast(dtype(ML, ML.date)).alias(ML.date),
            pl.col(G.home_club_id).alias(ML.home_club_id),
            pl.col(G.away_club_id).alias(ML.away_club_id),
            pl.col(G.home_club_goals).cast(dtype(ML, ML.home_goals)).alias(ML.home_goals),
            pl.col(G.away_club_goals).cast(dtype(ML, ML.away_goals)).alias(ML.away_goals),
            pl.col(G.season).alias(ML.season),
            pl.col(G.season).cast(dtype(ML, ML.season_start)).alias(ML.season_start),
            pl.col(G.round).alias(ML.round),
            pl.col(G.home_club_formation).alias(ML.home_formation),
            pl.col(G.away_club_formation).alias(ML.away_formation),
        )
        .filter(pl.col(ML.date).is_not_null())
    )
    return cast(DataFrame[ML], out)


def curry_match_join_match_w_league(
    matches_df: DataFrame[G],
) -> Morphism[DataFrame[C], DataFrame[ML]]:
    return lambda league_df: _join_match_w_league(matches_df, league_df)
