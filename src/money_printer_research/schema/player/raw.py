"""Raw Understat player match stats, every column as collected."""

import pandera.polars as pa
from pandera.typing import FieldType as F


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


PLAYER_STAT = UnderstatPlayerRaw


__all__ = ["PLAYER_STAT", "UnderstatPlayerRaw"]
