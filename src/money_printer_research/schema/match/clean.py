"""Cleaned matches: the Understat schedule, Transfermarkt games, and the two joined.

ScheduleCols       the cleaned Understat schedule (kept in memory)
MatchLeaguesCols   match_leagues.csv: Transfermarkt games, every competition
CleanedMatchCols   cleaned_matches.csv: each Understat match with its Transfermarkt game
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.competition.comp_type import COMP_TYPES
from money_printer_research.schema.match.formation import FormationCols


class ScheduleCols(pa.DataFrameModel):
    """Cleaned Understat schedule: played matches only, with goals and xG. Key: game_id."""

    league: F[str] = pa.Field(
        description=(
            'League in soccerdata form: "ENG-Premier League", "ESP-La Liga", "ITA-Serie A", '
            '"GER-Bundesliga", "FRA-Ligue 1".'
        )
    )

    season: F[int] = pa.Field(
        description=(
            "Season as start and end year, two digits each: 1415 is 2014/15. An int here, a "
            "str in UnderstatPlayerRaw."
        )
    )

    game_id: F[int] = pa.Field(
        description="Understat match id, unique. The join key to UnderstatPlayerRaw.game_id."
    )

    date: F[datetime.datetime] = pa.Field(
        description=(
            "Kick-off date and time as Understat lists it, no time zone. A few games have "
            "00:00 as the time, so use only the date part when comparing across sources."
        )
    )

    home_team: F[str] = pa.Field(
        description=(
            'Understat team name: "Arsenal", "Manchester United". Not the same spelling as '
            "other sources; norm_team_names maps it through ALIASES."
        )
    )

    away_team: F[str] = pa.Field(
        description="Understat team name of the away side; see home_team."
    )

    home_goals: F[int] = pa.Field(
        nullable=True, description="Home goals at full time. Null when is_result is False."
    )

    away_goals: F[int] = pa.Field(
        nullable=True, description="Away goals at full time. Null when is_result is False."
    )

    home_xg: F[float] = pa.Field(
        nullable=True,
        description="Home expected goals (sum of shot xG): 1.554. Null when is_result is False.",
    )

    away_xg: F[float] = pa.Field(
        nullable=True, description="Away expected goals: 0.158. Null when is_result is False."
    )

    is_result: F[bool] = pa.Field(
        description=(
            "True once the match has a result. False rows are fixtures never played, such as "
            "the Ligue 1 2019/20 games cancelled for COVID."
        )
    )

    season_start: F[pl.Int32] = pa.Field(
        description=(
            "Season start year from season: 1415 -> 2014. Every clean frame has it, so join"
            " on it rather than on season, whose format differs per source."
        )
    )


class MatchLeaguesCols(FormationCols):
    """match_leagues.csv: Transfermarkt games from clean_data.match_leagues.join_match_w_league.

    GameFormationCols (games.parquet after encode_formation) left-joined to the
    competitions on competition_id, renamed, with comp_type and the Understat league
    added. Inherits the encoded formation columns and their check from FormationCols.
    Every competition the clubs play, not only the leagues. Key: tm_game_id.
    """

    tm_game_id: F[str] = pa.Field(
        description='Transfermarkt game id (game_id in games.parquet): "2458308".'
    )

    competition_id: F[str] = pa.Field(
        description=(
            'Transfermarkt competition code: "GB1", "FAC", "CL"; see '
            "TransfermarktGamesRaw.competition_id."
        )
    )

    comp_type: F[str] = pa.Field(
        isin=COMP_TYPES,
        description=(
            "league: the five Understat leagues (LEAGUE in match_leagues). europe: UEFA club"
            " competitions and their qualifying (CL, CLQ, EL, ELQ, UCOL, ECLQ, USC), set by "
            "competition_id because Transfermarkt files EL and UCOL under type other. "
            "domestic_cup: Transfermarkt type domestic_cup, plus CGB (EFL Cup), which "
            "competitions.parquet does not list. other: domestic super cups (GBCS, SUC, "
            "SCI, DFL, FRCH) and KLUB (Club World Cup)."
        ),
    )

    league: F[str] = pa.Field(
        nullable=True,
        description=(
            'Understat league name for league games ("ENG-Premier League"), so they line up'
            " with the Understat schedule. Null for every other competition."
        ),
    )

    date: F[datetime.date] = pa.Field(description="Match date. Games without a date are dropped.")

    home_club_id: F[pl.Int32] = pa.Field(
        description=(
            "Transfermarkt club id of the home side; map_club_w_team maps it to the Understat"
            " team name."
        )
    )

    away_club_id: F[pl.Int32] = pa.Field(description="Transfermarkt club id of the away side.")

    home_goals: F[int] = pa.Field(
        description="Home goals in the official result (home_club_goals)."
    )

    away_goals: F[int] = pa.Field(
        description="Away goals in the official result (away_club_goals)."
    )

    season: F[str] = pa.Field(
        description='Season start year as a str, as Transfermarkt writes it: "2014".'
    )

    season_start: F[pl.Int32] = pa.Field(description="season as an integer: 2014.")

    round: F[str] = pa.Field(description='Round name: "1. Matchday", "First Round", "Final".')

    home_formation: F[str] = pa.Field(
        nullable=True,
        description=(
            'Home starting formation as Transfermarkt writes it: "4-3-3 Attacking". Null when'
            " Transfermarkt has no line-up. Encoded in home_shape, home_variant and the line"
            " counts (FormationCols)."
        ),
    )

    away_formation: F[str] = pa.Field(
        nullable=True, description="Away starting formation; see home_formation."
    )


class CleanedMatchCols(ScheduleCols, FormationCols):
    """cleaned_matches.csv: each Understat match with its Transfermarkt league game.

    Built by join_match_w_stat from the cleaned schedule (ScheduleCols).

    Left join on league, season_start, home_team and away_team (club ids mapped to
    teams through clubs.csv), not on the date: postponed or resumed games can sit days
    apart in the two sources. Same rows as the schedule, one per game_id; the Transfermarkt
    columns, the encoded formations of FormationCols included, are null where no game
    matched. Key: game_id.
    """

    tm_game_id: F[str] = pa.Field(
        nullable=True,
        description=(
            "Transfermarkt game id of the same fixture (match_leagues.csv tm_game_id). Null when "
            "no Transfermarkt league game matched."
        ),
    )

    tm_season: F[str] = pa.Field(
        nullable=True, description='Transfermarkt season: "2014". Null when unmatched.'
    )

    tm_round: F[str] = pa.Field(
        nullable=True, description='Transfermarkt round: "1. Matchday". Null when unmatched.'
    )

    home_formation: F[str] = pa.Field(
        nullable=True,
        description=(
            'Home starting formation from Transfermarkt: "4-3-3 Attacking". Null when '
            "unmatched or Transfermarkt has no line-up. Encoded in home_shape, home_variant "
            "and the line counts (FormationCols)."
        ),
    )

    away_formation: F[str] = pa.Field(
        nullable=True, description="Away starting formation; see home_formation."
    )

    tm_home_goals: F[int] = pa.Field(
        nullable=True,
        description=(
            "Home goals in Transfermarkt's official result. Differs from home_goals (the "
            "pitch score) for an overturned result. Null when unmatched."
        ),
    )

    tm_away_goals: F[int] = pa.Field(
        nullable=True,
        description="Away goals in Transfermarkt's official result; see tm_home_goals.",
    )


CLEANED_MATCH_STAT = ScheduleCols


MATCH_LEAGUE = MatchLeaguesCols


CLEANED_MATCH = CleanedMatchCols


__all__ = [
    "CLEANED_MATCH",
    "CLEANED_MATCH_STAT",
    "CleanedMatchCols",
    "MATCH_LEAGUE",
    "MatchLeaguesCols",
    "ScheduleCols",
]
