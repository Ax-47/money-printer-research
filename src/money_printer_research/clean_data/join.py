"""Join the exact features onto the matches: one row per Understat game_id."""

import polars as pl
from pandera.typing.polars import DataFrame

from ..cool_stuff import returns
from ..schema import CLEANED_MATCH, MATCH_FEATURES, EloCols, TeamFatigueCols, TeamScheduleCols

E = EloCols

# Elo has no game_id; a team plays at most once a day, so date and both teams name a game.
ELO_KEYS = (E.match_date, E.home_team, E.away_team)


@returns(MATCH_FEATURES)
def join_features(
    matches: DataFrame[CLEANED_MATCH],
    elo: DataFrame[EloCols],
    fatigue: DataFrame[TeamFatigueCols],
    schedule: DataFrame[TeamScheduleCols],
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Join pre-match Elo, team fatigue, schedule and player-slot features onto the matches.

    Left joins, each 1:1, so every match keeps its row: a feature is null where its
    source has no value for the game (fatigue and the player slots for the two games
    without Understat line-ups). players is get_player_features' output (PLAYER_SLOTS).
    """
    elo_cols = elo.select(*ELO_KEYS, E.full_time_result, E.elo_home, E.elo_away, E.elo_diff)
    return (
        matches.with_columns(pl.col(CLEANED_MATCH.date).dt.date().alias(E.match_date))
        .join(elo_cols, on=list(ELO_KEYS), how="left", validate="1:1")
        .join(fatigue, on=CLEANED_MATCH.game_id, how="left", validate="1:1")
        .join(schedule, on=CLEANED_MATCH.game_id, how="left", validate="1:1")
        .join(players, on=CLEANED_MATCH.game_id, how="left", validate="1:1")
    )


__all__ = ["ELO_KEYS", "join_features"]
