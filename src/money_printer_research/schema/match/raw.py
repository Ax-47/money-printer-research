"""Raw match files: the Understat schedule and Transfermarkt games.parquet, every column."""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F


class UnderstatScheduleRaw(pa.DataFrameModel):
    """data/raw/xg_schedule.csv (Understat read_schedule). Read with try_parse_dates.

    One row per league match in the five big leagues, played or not.
    Key: game_id. Unplayed fixtures have is_result False and null goals and xG.
    """

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

    game: F[str] = pa.Field(
        description=(
            'Readable match label "<date> <home>-<away>": "2014-08-16 Arsenal-Crystal Palace".'
        )
    )

    league_id: F[int] = pa.Field(
        description="Understat league id: 1 EPL, 2 Serie A, 3 Bundesliga, 4 La Liga, 5 Ligue 1."
    )

    season_id: F[int] = pa.Field(description="Season start year: 2014 for 1415.")

    game_id: F[int] = pa.Field(
        description="Understat match id, unique. The join key to UnderstatPlayerRaw.game_id."
    )

    date: F[datetime.datetime] = pa.Field(
        description=(
            "Kick-off date and time as Understat lists it, no time zone. A few games have "
            "00:00 as the time, so use only the date part when comparing across sources."
        )
    )

    home_team_id: F[int] = pa.Field(description="Understat team id of the home side.")

    away_team_id: F[int] = pa.Field(description="Understat team id of the away side.")

    home_team: F[str] = pa.Field(
        description=(
            'Understat team name: "Arsenal", "Manchester United". Not the same spelling as '
            "other sources; norm_team_names maps it through ALIASES."
        )
    )

    away_team: F[str] = pa.Field(
        description="Understat team name of the away side; see home_team."
    )

    away_team_code: F[str] = pa.Field(
        description=(
            'Three-letter team code: "CRY". Not unique across leagues ("MON" is Monaco and '
            "Monza), so do not join on it."
        )
    )

    home_team_code: F[str] = pa.Field(
        description=('Three-letter team code: "ARS"; see away_team_code.')
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

    has_data: F[bool] = pa.Field(
        description=(
            "True when Understat has shot and player data for the match. False for unplayed "
            "fixtures and for 2017-04-16 SC Bastia-Lyon (abandoned; 0-0 with 0.0 xG)."
        )
    )

    url: F[str] = pa.Field(
        description=('Match page: "https://understat.com/match/4755" (ends in game_id).')
    )


class TransfermarktGamesRaw(pa.DataFrameModel):
    """<transfermarkt.out_dir>/games.parquet (export of the Transfermarkt DuckDB).

    One row per game in the five leagues, their domestic cups and super cups, and
    UEFA competitions, so cup opponents from lower leagues appear too. Key: game_id.
    """

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

    home_club_position: F[pl.Int32] = pa.Field(
        nullable=True,
        description="Home side's league table position, 1 to 20. Null outside league games.",
    )

    away_club_position: F[pl.Int32] = pa.Field(
        nullable=True, description="Away side's league table position. Null outside league games."
    )

    home_club_manager_name: F[str] = pa.Field(
        nullable=True, description=('Home manager on the day: "Lucien Favre".')
    )

    away_club_manager_name: F[str] = pa.Field(
        nullable=True, description="Away manager on the day."
    )

    stadium: F[str] = pa.Field(nullable=True, description="Stadium name.")

    attendance: F[pl.Int32] = pa.Field(
        nullable=True, description="Spectators. Null when not reported, which is common."
    )

    referee: F[str] = pa.Field(nullable=True, description=('Referee name: "Patrick Ittrich".'))

    url: F[str] = pa.Field(nullable=True, description="Match report page, ending in game_id.")

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

    home_club_name: F[str] = pa.Field(
        nullable=True, description=('Transfermarkt club name: "Borussia Mönchengladbach".')
    )

    away_club_name: F[str] = pa.Field(
        nullable=True, description="Transfermarkt club name of the away side."
    )

    aggregate: F[str] = pa.Field(
        nullable=True,
        description=(
            'Score as "home:away": "1:3". Despite the name it is this game\'s score, not a '
            "two-leg total (always equal to the goals columns in the collected files)."
        ),
    )

    competition_type: F[str] = pa.Field(
        nullable=True,
        description=(
            "domestic_league, domestic_cup, international_cup or other; the same value as "
            "TransfermarktCompetitionsRaw.type. Null for CGB and KLUB."
        ),
    )


MATCH_STAT = UnderstatScheduleRaw


MATCH_INFO = TransfermarktGamesRaw


__all__ = ["MATCH_INFO", "MATCH_STAT", "TransfermarktGamesRaw", "UnderstatScheduleRaw"]
