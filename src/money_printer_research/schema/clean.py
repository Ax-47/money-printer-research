"""Clean single-source frames: each from one source, cleaned but not yet joined.

xg.csv (ScheduleCols), std_tm.csv (MatchLeaguesCols) and player_stat.csv (PlayerStatCols).
The Understat models keep the selected raw columns (schema/selected.py) with the
same descriptions and add season_start; MatchLeaguesCols is join_match_w_league's output.
Joined outputs are in schema/join.py.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.formation import FormationCols


class ScheduleCols(pa.DataFrameModel):
    """xg.csv: Understat schedule, played matches only, with goals and xG. Key: game_id."""

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
            "other sources; rename_team_names maps it through ALIASES."
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
    """std_tm.csv: Transfermarkt games from clean_data.match_leagues.join_match_w_league.

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
        isin=["league", "domestic_cup", "europe", "other"],
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
            " with xg.csv. Null for every other competition."
        ),
    )

    date: F[datetime.date] = pa.Field(description="Match date. Games without a date are dropped.")

    home_club_id: F[pl.Int32] = pa.Field(
        description=(
            "Transfermarkt club id of the home side; map_clubs maps it to the Understat team name."
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


class PlayerStatCols(pa.DataFrameModel):
    """player_stat.csv: Understat player match stats, one row per player per match.

    Key: (game_id, player_id).
    """

    league: F[str] = pa.Field(
        description=(
            'League in soccerdata form: "ENG-Premier League"; see UnderstatScheduleRaw.league.'
        )
    )

    season: F[str] = pa.Field(
        description=(
            'Season as start and end year, two digits each: "1415". A str here, an int in '
            "UnderstatScheduleRaw."
        )
    )

    game: F[str] = pa.Field(
        description=('Match label "<date> <home>-<away>", same as UnderstatScheduleRaw.game.')
    )

    team: F[str] = pa.Field(
        description=(
            "Understat name of the player's team: \"Manchester City\". Equals the schedule's "
            "home_team or away_team for the match."
        )
    )

    player: F[str] = pa.Field(
        description=('Player name as Understat writes it: "Alexis Sánchez".')
    )

    game_id: F[int] = pa.Field(
        description="Understat match id; joins to UnderstatScheduleRaw.game_id."
    )

    team_id: F[int] = pa.Field(
        description="Understat team id; same ids as the schedule's home_team_id and away_team_id."
    )

    player_id: F[int] = pa.Field(description="Understat player id, stable across seasons.")

    position: F[str] = pa.Field(
        description=(
            "Position in the line-up: GK, DR, DC, DL, DMR, DML, DMC, MR, MC, ML, AMR, AMC, "
            'AML, FWR, FW, FWL; "Sub" for a substitute. Starters are every non-"Sub" row.'
        )
    )

    minutes: F[int] = pa.Field(description="Minutes played, at most 90.")

    xg: F[float] = pa.Field(description="Expected goals from the player's shots.")

    xa: F[float] = pa.Field(
        description="Expected assists: xG of the shots that came from the player's passes."
    )

    goals: F[int] = pa.Field(description="Goals scored, own goals excluded.")

    own_goals: F[int] = pa.Field(description="Own goals scored.")

    shots: F[int] = pa.Field(description="Shots taken.")

    xg_chain: F[float] = pa.Field(
        description="xG of every possession the player touched that ended in a shot."
    )

    xg_buildup: F[float] = pa.Field(
        description=(
            "xg_chain without the possessions where the player shot or made the key pass, so "
            "never above xg_chain."
        )
    )

    assists: F[int] = pa.Field(description="Assists.")

    key_passes: F[int] = pa.Field(description="Passes that led directly to a shot.")

    yellow_cards: F[int] = pa.Field(description="Yellow cards, 0 or 1.")

    red_cards: F[int] = pa.Field(description="Red cards, 0 or 1.")

    season_start: F[pl.Int32] = pa.Field(
        description='Season start year from season: "1415" -> 2014.'
    )


CLEANED_MATCH_STAT = ScheduleCols
MATCH_LEAGUE = MatchLeaguesCols
CLEANED_PLAYER_STAT = PlayerStatCols

__all__ = [
    "CLEANED_MATCH_STAT",
    "MATCH_LEAGUE",
    "CLEANED_PLAYER_STAT",
    "MatchLeaguesCols",
    "PlayerStatCols",
    "ScheduleCols",
]
