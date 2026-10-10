"""Understat schedule + Transfermarkt league games -> cleaned_matches (CleanedMatchCols)."""

import polars as pl
from pandera.typing.polars import DataFrame

from ..cool_stuff import returns
from ..schema import CleanedMatchCols
from ..schema import ScheduleCols
from ..schema import ClubMapCols
from ..schema import FORMATION_COLUMNS
from ..schema import MatchLeaguesCols


@returns(CleanedMatchCols)
def join_match_w_stat(
    tm: DataFrame[MatchLeaguesCols],
    clubs: DataFrame[ClubMapCols],
    schedule: DataFrame[ScheduleCols],
) -> pl.DataFrame:
    """Understat schedule rows with their Transfermarkt league game (left join).

    Keyed on league, season and both teams rather than the date: a fixture is
    played once per season, and postponed or resumed games can sit days apart
    in the two sources (Udinese-Roma 2024: 14 April vs 25 April). Transfermarkt
    columns get a tm_ prefix so Understat's goals and date keep their names; the
    encoded formation columns (FormationCols) keep theirs, which only match_leagues has.
    """
    club_team = clubs.filter(~pl.col(ClubMapCols.duplicate)).select(
        ClubMapCols.club_id, ClubMapCols.team
    )
    tm_league = (
        tm.filter(pl.col(MatchLeaguesCols.comp_type) == "league")
        .join(
            club_team.rename(
                {
                    ClubMapCols.club_id: MatchLeaguesCols.home_club_id,
                    ClubMapCols.team: ScheduleCols.home_team,
                }
            ),
            on=MatchLeaguesCols.home_club_id,
        )
        .join(
            club_team.rename(
                {
                    ClubMapCols.club_id: MatchLeaguesCols.away_club_id,
                    ClubMapCols.team: ScheduleCols.away_team,
                }
            ),
            on=MatchLeaguesCols.away_club_id,
        )
        .select(
            MatchLeaguesCols.league,
            MatchLeaguesCols.season_start,
            ScheduleCols.home_team,
            ScheduleCols.away_team,
            pl.col(MatchLeaguesCols.tm_game_id).alias(CleanedMatchCols.tm_game_id),
            pl.col(MatchLeaguesCols.season).alias(CleanedMatchCols.tm_season),
            pl.col(MatchLeaguesCols.round).alias(CleanedMatchCols.tm_round),
            pl.col(MatchLeaguesCols.home_formation).alias(CleanedMatchCols.home_formation),
            pl.col(MatchLeaguesCols.away_formation).alias(CleanedMatchCols.away_formation),
            pl.col(MatchLeaguesCols.home_goals).alias(CleanedMatchCols.tm_home_goals),
            pl.col(MatchLeaguesCols.away_goals).alias(CleanedMatchCols.tm_away_goals),
            pl.col(*FORMATION_COLUMNS),  # same names in L and CM, both from FormationCols
        )
    )
    # L.league / L.season_start have the same names as in S.
    keys = [
        ScheduleCols.league,
        ScheduleCols.season_start,
        ScheduleCols.home_team,
        ScheduleCols.away_team,
    ]
    return schedule.join(tm_league, on=keys, how="left", validate="1:1")
