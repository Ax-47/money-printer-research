"""Understat schedule + Transfermarkt league games -> stat_matches (StatMatchCols)."""

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import CLEANED_MATCH as CM
from money_printer_research.schema import CLEANED_MATCH_STAT as S
from money_printer_research.schema import CLUB_MAP as C
from money_printer_research.schema import FORMATION_COLUMNS
from money_printer_research.schema import MATCH_LEAGUE as L


@returns(CM)
def join_match_w_stat(
    tm: DataFrame[L], clubs: DataFrame[C], schedule: DataFrame[S]
) -> pl.DataFrame:
    """Understat schedule rows with their Transfermarkt league game (left join).

    Keyed on league, season and both teams rather than the date: a fixture is
    played once per season, and postponed or resumed games can sit days apart
    in the two sources (Udinese-Roma 2024: 14 April vs 25 April). Transfermarkt
    columns get a tm_ prefix so Understat's goals and date keep their names; the
    encoded formation columns (FormationCols) keep theirs, which only std_tm has.
    """
    club_team = clubs.filter(~pl.col(C.duplicate)).select(C.club_id, C.team)
    tm_league = (
        tm.filter(pl.col(L.comp_type) == "league")
        .join(
            club_team.rename({C.club_id: L.home_club_id, C.team: S.home_team}), on=L.home_club_id
        )
        .join(
            club_team.rename({C.club_id: L.away_club_id, C.team: S.away_team}), on=L.away_club_id
        )
        .select(
            L.league,
            L.season_start,
            S.home_team,
            S.away_team,
            pl.col(L.tm_game_id).alias(CM.tm_game_id),
            pl.col(L.season).alias(CM.tm_season),
            pl.col(L.round).alias(CM.tm_round),
            pl.col(L.home_formation).alias(CM.home_formation),
            pl.col(L.away_formation).alias(CM.away_formation),
            pl.col(L.home_goals).alias(CM.tm_home_goals),
            pl.col(L.away_goals).alias(CM.tm_away_goals),
            pl.col(*FORMATION_COLUMNS),  # same names in L and CM, both from FormationCols
        )
    )
    # L.league / L.season_start have the same names as in S.
    keys = [S.league, S.season_start, S.home_team, S.away_team]
    return schedule.join(tm_league, on=keys, how="left", validate="1:1")
