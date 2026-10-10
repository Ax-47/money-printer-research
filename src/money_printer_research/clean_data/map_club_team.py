import polars as pl
from pandera.typing.polars import DataFrame

from ..cool_stuff import Err, Ok, Result, monad, returns
from ..schema import ScheduleCols
from ..schema import ClubMapCols
from ..schema import MatchLeaguesCols

# A mapping is doubtful when two club ids share a team, or when an established
# club (enough games) has few fixtures backing its pairing.
MIN_COVERAGE = 0.9
MIN_GAMES_FOR_COVERAGE = 10


def _league_games_per_club(tm_league: pl.DataFrame) -> pl.DataFrame:
    """club_id -> number of Transfermarkt league games (C.games)."""
    return (
        pl.concat(
            [
                tm_league.select(pl.col(MatchLeaguesCols.home_club_id).alias(ClubMapCols.club_id)),
                tm_league.select(pl.col(MatchLeaguesCols.away_club_id).alias(ClubMapCols.club_id)),
            ]
        )
        .group_by(ClubMapCols.club_id)
        .len(ClubMapCols.games)
    )


def _same_fixture_pairs(
    tm_league: pl.DataFrame, schedule: DataFrame[ScheduleCols], max_day_shift: int
) -> pl.DataFrame:
    """(club_id, team) once per side of every fixture both sources agree on.

    Fixtures agree when league and score match and the dates are at most
    max_day_shift days apart.
    """
    us = schedule.select(
        ScheduleCols.league,
        pl.col(ScheduleCols.date).cast(pl.Date).alias("us_date"),
        ScheduleCols.home_team,
        ScheduleCols.away_team,
        ScheduleCols.home_goals,
        ScheduleCols.away_goals,
    )
    # L.league / L.home_goals / L.away_goals have the same names as in S.
    fixtures = tm_league.join(
        us, on=[MatchLeaguesCols.league, MatchLeaguesCols.home_goals, MatchLeaguesCols.away_goals]
    ).filter(
        (pl.col(MatchLeaguesCols.date) - pl.col("us_date")).dt.total_days().abs() <= max_day_shift
    )
    return pl.concat(
        [
            fixtures.select(
                pl.col(MatchLeaguesCols.home_club_id).alias(ClubMapCols.club_id),
                pl.col(ScheduleCols.home_team).alias(ClubMapCols.team),
            ),
            fixtures.select(
                pl.col(MatchLeaguesCols.away_club_id).alias(ClubMapCols.club_id),
                pl.col(ScheduleCols.away_team).alias(ClubMapCols.team),
            ),
        ]
    )


@returns(ClubMapCols)
def _map_clubs(
    tm: DataFrame[MatchLeaguesCols], schedule: DataFrame[ScheduleCols], max_day_shift: int = 1
) -> pl.DataFrame:
    """Transfermarkt club_id -> Understat team, learned from league fixtures.

    Each club gets the team it was paired with most often. coverage: share of the
    club's Transfermarkt league games that back the pairing (close to 1.0 when
    correct). duplicate: two club ids map to one team.
    """
    tm_league = tm.filter(pl.col(MatchLeaguesCols.comp_type) == "league")
    pairs = _same_fixture_pairs(tm_league, schedule, max_day_shift)
    return (
        pairs.group_by(ClubMapCols.club_id, ClubMapCols.team)
        .len()
        .sort("len", descending=True)
        .group_by(ClubMapCols.club_id, maintain_order=True)
        .agg(pl.col(ClubMapCols.team).first(), pl.col("len").first().alias(ClubMapCols.n))
        .join(_league_games_per_club(tm_league), on=ClubMapCols.club_id)
        .with_columns(
            (pl.col(ClubMapCols.n) / pl.col(ClubMapCols.games))
            .round(3)
            .alias(ClubMapCols.coverage)
        )
        .with_columns(pl.len().over(ClubMapCols.team).gt(1).alias(ClubMapCols.duplicate))
        .sort(ClubMapCols.team)
    )


@monad
def map_club_w_team(
    tm: DataFrame[MatchLeaguesCols], schedule: DataFrame[ScheduleCols], max_day_shift: int = 1
) -> Result[DataFrame[ClubMapCols]]:
    """_map_clubs, as an Err when there is nothing to map or a pairing is doubtful.

    Nothing to map usually means match_leagues has no league games (comp_type); a doubtful
    pairing usually means a missing ALIASES entry.
    """
    league_games = tm.filter(pl.col(MatchLeaguesCols.comp_type) == "league").height
    if league_games == 0:
        return Err("map_club_w_team: match_leagues has no league games (check comp_type)")

    clubs = _map_clubs(tm, schedule, max_day_shift).if_err()
    if clubs.height == 0:
        return Err(
            f"map_club_w_team: no club matched any of {league_games:,} league games "
            "(check league names and dates between the two sources)"
        )

    doubtful = clubs.filter(
        pl.col(ClubMapCols.duplicate)
        | (
            (pl.col(ClubMapCols.coverage) < MIN_COVERAGE)
            & (pl.col(ClubMapCols.games) >= MIN_GAMES_FOR_COVERAGE)
        )
    )
    if doubtful.height:
        return Err(f"map_club_w_team: doubtful club mappings (check ALIASES):\n{doubtful}")
    return Ok(clubs)
