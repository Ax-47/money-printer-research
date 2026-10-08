"""The raw columns the clean pipeline reads, one model per raw file.

Each class lists only the columns of its raw model (schema/raw.py) that the
pipeline uses: same annotation, nullability and description, copied line for line.
Collection still checks every raw column. tests/test_selected.py checks that every
column here matches its raw column.

Columns are F[...] (pandera FieldType), so pl.col(UnderstatSchedule.season) works
and pyright types the attribute as str.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.raw import LEAGUE_INFO, MATCH_INFO, MATCH_STAT, PS


class UnderstatSchedule(pa.DataFrameModel):
    """The Understat schedule columns the clean pipeline reads."""

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


class UnderstatPlayer(pa.DataFrameModel):
    """The Understat player match stat columns the clean pipeline reads."""

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

    goals: F[int] = pa.Field(description="Goals scored, own goals excluded.")

    own_goals: F[int] = pa.Field(description="Own goals scored.")

    shots: F[int] = pa.Field(description="Shots taken.")

    xg: F[float] = pa.Field(description="Expected goals from the player's shots.")

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

    xa: F[float] = pa.Field(
        description="Expected assists: xG of the shots that came from the player's passes."
    )

    key_passes: F[int] = pa.Field(description="Passes that led directly to a shot.")

    yellow_cards: F[int] = pa.Field(description="Yellow cards, 0 or 1.")

    red_cards: F[int] = pa.Field(description="Red cards, 0 or 1.")


class TransfermarktGames(pa.DataFrameModel):
    """The Transfermarkt game columns the clean pipeline reads."""

    game_id: F[str] = pa.Field(description=('Transfermarkt game id as a str: "2458308".'))

    competition_id: F[str] = pa.Field(
        description=(
            "Transfermarkt competition code: league GB1, ES1, IT1, L1, FR1; cups FAC, CDR, "
            "CIT, DFB, CGB; UEFA CL, EL, UCOL and their qualifying CLQ, ELQ, ECLQ; super cups"
            " GBCS, SUC, SCI, DFL, FRCH, USC; KLUB. Joins to TransfermarktCompetitionsRaw, "
            "except CGB and KLUB, which that file does not list."
        )
    )

    season: F[str] = pa.Field(description=('Season start year as a str: "2014" for 2014/15.'))

    round: F[str] = pa.Field(
        nullable=True,
        description=(
            'Round name: "1. Matchday" in leagues, "First Round", "Final" and so on in cups.'
        ),
    )

    date: F[datetime.date] = pa.Field(nullable=True, description="Match date, no time.")

    home_club_id: F[pl.Int32] = pa.Field(
        description=(
            "Transfermarkt club id of the home side. map_clubs maps it to the Understat team name."
        )
    )

    away_club_id: F[pl.Int32] = pa.Field(description="Transfermarkt club id of the away side.")

    home_club_goals: F[pl.Int32] = pa.Field(
        nullable=True,
        description=(
            "Home goals in the official result, which can differ from the pitch score "
            "Understat records (an overturned result)."
        ),
    )

    away_club_goals: F[pl.Int32] = pa.Field(
        nullable=True, description="Away goals in the official result; see home_club_goals."
    )

    home_club_formation: F[str] = pa.Field(
        nullable=True,
        description=(
            'Home starting formation: "4-2-3-1", "4-3-3 Attacking", "4-4-2 double 6", "3-5-2 '
            'flat". Null when Transfermarkt has no line-up.'
        ),
    )

    away_club_formation: F[str] = pa.Field(
        nullable=True, description="Away starting formation; see home_club_formation."
    )


class TransfermarktCompetitions(pa.DataFrameModel):
    """The Transfermarkt competition columns the clean pipeline reads."""

    competition_id: F[str] = pa.Field(
        description=('Transfermarkt competition code: "GB1", "CL"; see TransfermarktGamesRaw.')
    )

    type: F[str] = pa.Field(
        nullable=True,
        description=(
            "domestic_league, domestic_cup, international_cup, national_team_competition or "
            "other. Transfermarkt files the Europa League, Conference League and domestic "
            'super cups under "other"; only the Champions League, its qualifying and the UEFA'
            " Super Cup are international_cup."
        ),
    )


# Raw model -> the model of the columns the clean pipeline reads.
# Collection checks the raw model (every column); when the pipeline starts using a
# column, copy its block from the raw model into the selected one.
SELECTED: dict[type[pa.DataFrameModel], type[pa.DataFrameModel]] = {
    MATCH_STAT: UnderstatSchedule,
    PS: UnderstatPlayer,
    MATCH_INFO: TransfermarktGames,
    LEAGUE_INFO: TransfermarktCompetitions,
}


__all__ = [
    "SELECTED",
    "TransfermarktCompetitions",
    "TransfermarktGames",
    "UnderstatPlayer",
    "UnderstatSchedule",
]
