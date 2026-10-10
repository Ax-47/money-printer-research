"""Pre-match fatigue features per player, and per-team summaries for matches.

All load features use only matches before the current one (closed="left"),
so they are known at kick-off. Understat covers league matches only, so rest
days are overstated for clubs that also play cups or European games.
"""

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import (
    CleanedMatchCols,
    PlayerStatCols,
    FatigueInputCols,
    MatchSideCols,
    PlayerFatigueCols,
    TeamFatigueCols,
)


FATIGUE_COLS = [
    PlayerFatigueCols.rest_days,
    PlayerFatigueCols.n_7d,
    PlayerFatigueCols.min_7d,
    PlayerFatigueCols.min_28d,
    PlayerFatigueCols.streak_prev,
    PlayerFatigueCols.fatigue_score,
]

# A player's run of consecutive games belongs to one team in one season.
_STREAK_KEYS = (
    FatigueInputCols.player_id,
    FatigueInputCols.season_start,
    FatigueInputCols.team_id,
)


def player_matches(
    player_stat: DataFrame[PlayerStatCols], matches: DataFrame[CleanedMatchCols]
) -> DataFrame[FatigueInputCols]:
    """player_stat with each match's date and the player's side, one row per player per match.

    player_stat has no date or side, so both come from the match: match_date is the
    date part of the kick-off, and a player is home when his team is the home_team.
    Duplicate (game_id, player_id) rows, which Understat repeats for a few games, are dropped.
    """
    match_info: DataFrame[MatchSideCols] = MatchSideCols.validate(
        matches.select(
            pl.col(CleanedMatchCols.game_id).alias(MatchSideCols.game_id),
            pl.col(CleanedMatchCols.date).dt.date().alias(MatchSideCols.match_date),
            pl.col(CleanedMatchCols.home_team).alias(MatchSideCols.home_team),
        )
    )
    return FatigueInputCols.validate(
        player_stat.unique(
            [PlayerStatCols.game_id, PlayerStatCols.player_id], keep="first", maintain_order=True
        )
        .join(match_info, on=PlayerStatCols.game_id, how="inner", validate="m:1")
        .with_columns(
            (pl.col(PlayerStatCols.team) == pl.col(MatchSideCols.home_team)).alias(
                FatigueInputCols.is_home
            )
        )
        .drop(MatchSideCols.home_team)
    )


def _prev_window(col: pl.Expr, window: str) -> pl.Expr:
    """Sum of `col` over the previous `window`, excluding the current match."""
    return (
        col.rolling_sum_by(FatigueInputCols.match_date, window_size=window, closed="left")
        .over(FatigueInputCols.player_id)
        .fill_null(0)
    )


TEAM_GAME_NO = "_team_game_no"
NEW_STREAK = "_new_streak"
STREAK_ID = "_streak_id"


def _team_game_index(pm: DataFrame[FatigueInputCols]) -> pl.DataFrame:
    """Order of each game within a team's season: 0, 1, 2, ..."""
    return (
        pm.select(
            FatigueInputCols.team_id,
            FatigueInputCols.season_start,
            FatigueInputCols.game_id,
            FatigueInputCols.match_date,
        )
        .unique()
        .sort(FatigueInputCols.team_id, FatigueInputCols.match_date)
        .with_columns(
            pl.int_range(pl.len())
            .over(FatigueInputCols.team_id, FatigueInputCols.season_start)
            .alias(TEAM_GAME_NO)
        )
        .select(FatigueInputCols.team_id, FatigueInputCols.game_id, TEAM_GAME_NO)
    )


def _fatigue_score() -> pl.Expr:
    """Fatigue from 0 to 100, from the load columns. Hand-set weights, not fitted.

    Each term is scaled to 0-1: minutes in the last 7 days (180 is full),
    short rest, games in a row, and the acute:chronic workload ratio above 1.
    """
    chronic_week = pl.when(pl.col(PlayerFatigueCols.min_28d) > 0).then(
        pl.col(PlayerFatigueCols.min_28d) / 4
    )
    acwr = (pl.col(PlayerFatigueCols.min_7d) / chronic_week).clip(0, 3).fill_null(1)
    return (
        0.35 * (pl.col(PlayerFatigueCols.min_7d) / 180).clip(0, 1)
        + 0.25 * ((7 - pl.col(PlayerFatigueCols.rest_days).fill_null(7)) / 5).clip(0, 1)
        + 0.20 * (pl.col(PlayerFatigueCols.streak_prev) / 6).clip(0, 1)
        + 0.20 * (acwr - 1).clip(0, 1)
    ) * 100


def player_fatigue(player_matches: DataFrame[FatigueInputCols]) -> DataFrame[PlayerFatigueCols]:
    """player_matches plus pre-match fatigue columns (see FATIGUE_COLS)."""
    minutes = pl.col(FatigueInputCols.minutes)
    played = minutes.is_not_null().cast(pl.Int64)
    pm = (
        player_matches.join(
            _team_game_index(player_matches),
            on=[FatigueInputCols.team_id, FatigueInputCols.game_id],
            validate="m:1",
        )
        .sort(FatigueInputCols.player_id, FatigueInputCols.match_date)
        .with_columns(
            pl.col(FatigueInputCols.match_date)
            .diff()
            .dt.total_days()
            .over(FatigueInputCols.player_id)
            .alias(PlayerFatigueCols.rest_days),
            _prev_window(minutes, "7d").alias(PlayerFatigueCols.min_7d),
            _prev_window(played, "7d").alias(PlayerFatigueCols.n_7d),
            _prev_window(minutes, "28d").alias(PlayerFatigueCols.min_28d),
            # A new streak starts when the player missed the team's previous game,
            # or at his first game for a team (a transfer starts a new run).
            (pl.col(TEAM_GAME_NO).diff().over(_STREAK_KEYS) != 1)
            .fill_null(True)
            .alias(NEW_STREAK),
        )
        .with_columns(
            pl.col(NEW_STREAK).cast(pl.Int32).cum_sum().over(_STREAK_KEYS).alias(STREAK_ID)
        )
        .with_columns(
            pl.int_range(pl.len())
            .over(*_STREAK_KEYS, STREAK_ID)
            .alias(PlayerFatigueCols.streak_prev)
        )
    )
    out = pm.with_columns(_fatigue_score().round(1).alias(PlayerFatigueCols.fatigue_score)).drop(
        TEAM_GAME_NO, NEW_STREAK, STREAK_ID
    )
    return PlayerFatigueCols.validate(out, lazy=True)


def team_fatigue(fatigue: DataFrame[PlayerFatigueCols]) -> DataFrame[TeamFatigueCols]:
    """Mean starter fatigue per side, one row per game_id.

    A side with no starters recorded gets null in all of its columns, not 0.
    """
    starters = fatigue.filter(pl.col(PlayerFatigueCols.position) != "Sub")
    home, away = pl.col(PlayerFatigueCols.is_home), ~pl.col(PlayerFatigueCols.is_home)
    score, min_7d = pl.col(PlayerFatigueCols.fatigue_score), pl.col(PlayerFatigueCols.min_7d)
    out = (
        starters.group_by(PlayerFatigueCols.game_id)
        .agg(
            score.filter(home).mean().alias(TeamFatigueCols.h_fatigue),
            score.filter(away).mean().alias(TeamFatigueCols.a_fatigue),
            pl.when(home.any()).then(min_7d.filter(home).sum()).alias(TeamFatigueCols.h_min_7d),
            pl.when(away.any()).then(min_7d.filter(away).sum()).alias(TeamFatigueCols.a_min_7d),
        )
        .with_columns(
            (pl.col(TeamFatigueCols.h_fatigue) - pl.col(TeamFatigueCols.a_fatigue)).alias(
                TeamFatigueCols.fatigue_diff
            )
        )
    )
    return TeamFatigueCols.validate(out, lazy=True)


@returns(TeamFatigueCols)
def get_fatigue(
    player_stat: DataFrame[PlayerStatCols], matches: DataFrame[CleanedMatchCols]
) -> pl.DataFrame:
    """Pre-match team fatigue per game_id (team_fatigue), from player_stat and the matches."""
    return team_fatigue(player_fatigue(player_matches(player_stat, matches)))


__all__ = ["FATIGUE_COLS", "get_fatigue", "player_fatigue", "player_matches", "team_fatigue"]
