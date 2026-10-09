"""Column names and dtypes of the raw source files, as pandera DataFrameModels.

Every column is annotated FieldType[dtype] (F[...]), so class access is the column
name as a str both at runtime and for pyright, with no str() needed:

    pl.col(MATCH_STAT.season)              # "season"
    MATCH_STAT.validate(df, lazy=True)     # every missing column, wrong dtype and null

Each column's pa.Field(description=...) is its documentation; it is kept in the
schema (MATCH_STAT.to_schema().columns["season"].description). Example values come
from the files as collected for 2014/15 to 2025/26.

Raw names are kept as they come from each source; snake_columns_pl turns them
into the clean names used in schema.py.

Collection checks every column of these models (check_collected), so a source
that changes shape fails at download time. The columns the clean pipeline reads
are in schema/selected.py.
"""

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
            "other sources; rename_team_names maps it through ALIASES."
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


class UnderstatPlayerRaw(pa.DataFrameModel):
    """data/raw/player_stats/<league>/<season>.parquet (Understat read_player_match_stats).

    One row per player per match: the 11 starters of each side plus every substitute
    who came on. Key: (game_id, player_id).
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

    league_id: F[str] = pa.Field(
        description=('Understat league id as a str ("1"); an int in UnderstatScheduleRaw.')
    )

    season_id: F[int] = pa.Field(description="Season start year: 2014.")

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

    position_id: F[int] = pa.Field(
        description=('Number for position, 1 (GK) to 16 (FWL) in the order above; 17 is "Sub".')
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


class TransfermarktCompetitionsRaw(pa.DataFrameModel):
    """<transfermarkt.out_dir>/competitions.parquet (export of the Transfermarkt DuckDB).

    Lookup table, one row per competition. Key: competition_id.
    """

    competition_id: F[str] = pa.Field(
        description=('Transfermarkt competition code: "GB1", "CL"; see TransfermarktGamesRaw.')
    )

    competition_code: F[str] = pa.Field(
        nullable=True, description=('URL slug: "premier-league", "uefa-champions-league".')
    )

    name: F[str] = pa.Field(
        nullable=True,
        description="Same slug as competition_code in this export, not a display name.",
    )

    sub_type: F[str] = pa.Field(
        nullable=True,
        description=(
            "Finer type: first_tier, domestic_cup, domestic_super_cup, play_off, or the "
            "competition itself for international ones (uefa_europa_league, world_cup)."
        ),
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

    country_id: F[pl.Int32] = pa.Field(
        nullable=True, description="Transfermarkt country id; -1 for international competitions."
    )

    country_name: F[str] = pa.Field(
        nullable=True, description=('Country: "England". Null for international competitions.')
    )

    domestic_league_code: F[str] = pa.Field(
        nullable=True,
        description=(
            'competition_id of the country\'s top league: "GB1" for the FA Cup. Null for '
            "international competitions."
        ),
    )

    confederation: F[str] = pa.Field(
        nullable=True,
        description=(
            'Region name as Transfermarkt writes it: "europa", "asien", "afrika", "amerika", '
            '"fifa".'
        ),
    )

    total_clubs: F[pl.Int32] = pa.Field(
        nullable=True,
        description=(
            "Clubs in the country's top league (20 for every English competition), not in "
            "this competition. Null for international competitions."
        ),
    )

    url: F[str] = pa.Field(nullable=True, description="Competition page on transfermarkt.co.uk.")


MATCH_STAT = UnderstatScheduleRaw
PLAYER_STAT = UnderstatPlayerRaw
MATCH_INFO = TransfermarktGamesRaw
LEAGUE_INFO = TransfermarktCompetitionsRaw

type RawSchema = (
    type[UnderstatScheduleRaw]
    | type[UnderstatPlayerRaw]
    | type[TransfermarktGamesRaw]
    | type[TransfermarktCompetitionsRaw]
)


__all__ = ["LEAGUE_INFO", "MATCH_INFO", "MATCH_STAT", "PLAYER_STAT", "RawSchema"]
