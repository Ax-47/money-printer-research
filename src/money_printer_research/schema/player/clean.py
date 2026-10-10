"""player_stat.csv: Understat player match stats, one row per player per match."""

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F


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


__all__ = ["PlayerStatCols"]
