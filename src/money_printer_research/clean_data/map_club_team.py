import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import Err, Ok, Result, monad, returns
from money_printer_research.schema import CLEANED_MATCH_STAT as S
from money_printer_research.schema import CLUB_MAP as C
from money_printer_research.schema import MATCH_LEAGUE as L

# A mapping is doubtful when two club ids share a team, or when an established
# club (enough games) has few fixtures backing its pairing.
MIN_COVERAGE = 0.9
MIN_GAMES_FOR_COVERAGE = 10


@returns(C)
def _map_clubs(tm: DataFrame[L], schedule: DataFrame[S], max_day_shift: int = 1) -> pl.DataFrame:
    """Transfermarkt club_id -> Understat team, learned from league fixtures.

    coverage: share of the club's Transfermarkt league games that back the pairing
              (close to 1.0 when correct). duplicate: two club ids map to one team.
    """
    tm_league = tm.filter(pl.col(L.comp_type) == "league")
    us = schedule.select(
        S.league,
        pl.col(S.date).cast(pl.Date).alias("us_date"),
        S.home_team,
        S.away_team,
        S.home_goals,
        S.away_goals,
    )
    # L.league / L.home_goals / L.away_goals have the same names as in S.
    pairs = tm_league.join(us, on=[L.league, L.home_goals, L.away_goals]).filter(
        (pl.col(L.date) - pl.col("us_date")).dt.total_days().abs() <= max_day_shift
    )
    long = pl.concat(
        [
            pairs.select(
                pl.col(L.home_club_id).alias(C.club_id), pl.col(S.home_team).alias(C.team)
            ),
            pairs.select(
                pl.col(L.away_club_id).alias(C.club_id), pl.col(S.away_team).alias(C.team)
            ),
        ]
    )
    games = (
        pl.concat(
            [
                tm_league.select(pl.col(L.home_club_id).alias(C.club_id)),
                tm_league.select(pl.col(L.away_club_id).alias(C.club_id)),
            ]
        )
        .group_by(C.club_id)
        .len(C.games)
    )
    return (
        long.group_by(C.club_id, C.team)
        .len()
        .sort("len", descending=True)
        .group_by(C.club_id, maintain_order=True)
        .agg(pl.col(C.team).first(), pl.col("len").first().alias(C.n))
        .join(games, on=C.club_id)
        .with_columns((pl.col(C.n) / pl.col(C.games)).round(3).alias(C.coverage))
        .with_columns(pl.len().over(C.team).gt(1).alias(C.duplicate))
        .sort(C.team)
    )


@monad
def map_club_w_team(
    tm: DataFrame[L], schedule: DataFrame[S], max_day_shift: int = 1
) -> Result[DataFrame[C]]:
    """_map_clubs, as an Err when there is nothing to map or a pairing is doubtful.

    Nothing to map usually means std_tm has no league games (comp_type); a doubtful
    pairing usually means a missing ALIASES entry.
    """
    league_games = tm.filter(pl.col(L.comp_type) == "league").height
    if league_games == 0:
        return Err("map_club_w_team: std_tm has no league games (check comp_type)")

    clubs = _map_clubs(tm, schedule, max_day_shift).if_err()
    if clubs.height == 0:
        return Err(
            f"map_club_w_team: no club matched any of {league_games:,} league games "
            "(check league names and dates between the two sources)"
        )

    doubtful = clubs.filter(
        pl.col(C.duplicate)
        | ((pl.col(C.coverage) < MIN_COVERAGE) & (pl.col(C.games) >= MIN_GAMES_FOR_COVERAGE))
    )
    if doubtful.height:
        return Err(f"map_club_w_team: doubtful club mappings (check ALIASES):\n{doubtful}")
    return Ok(clubs)
