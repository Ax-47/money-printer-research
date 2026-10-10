"""The columns of the raw match files the clean pipeline reads."""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F


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
            "Transfermarkt club id of the home side. map_club_w_team maps it to the Understat"
            " team name."
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


__all__ = ["TransfermarktGames", "UnderstatSchedule"]
