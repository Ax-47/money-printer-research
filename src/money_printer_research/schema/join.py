"""Joined outputs: frames built from more than one clean frame.

    player_matches.csv  PlayerMatchCols
    matches.csv         MatchCols, plus the 88 player slot columns from slot_columns()
    clubs.csv           ClubMapCols
    stat_matches.csv    StatMatchCols

The single-source frames they are built from are in schema/clean.py.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.clean import PlayerStatCols, ScheduleCols
from money_printer_research.schema.formation import FormationCols

PLAYER_SLOT_FEATURES = ("xg90", "xa90", "form90", "prev_min", "fatigue_score")
N_SLOTS = 11


class PlayerMatchCols(PlayerStatCols):
    """player_matches.csv: player_stat plus match context, pre-match form and fatigue.

    Key: (game_id, player_id).
    """

    # Match context (_build_player_matches)
    match_date: F[datetime.date] = pa.Field(
        description="Match date, the date part of xg.csv's date."
    )
    home_team: F[str] = pa.Field(description="Home side of the match, after rename_team_names.")
    away_team: F[str] = pa.Field(description="Away side of the match, after rename_team_names.")
    is_home: F[bool] = pa.Field(description="True when team is home_team.")
    opponent: F[str] = pa.Field(
        description="The other side: away_team when is_home, otherwise home_team."
    )

    # Pre-match form (build_player.player_form); only matches before this one count
    prev_xg: F[float] = pa.Field(
        description="Player's xG accumulated before this match; window in player_form."
    )
    prev_xa: F[float] = pa.Field(
        description="Player's xA accumulated before this match; window in player_form."
    )
    prev_min: F[int] = pa.Field(
        description="Minutes accumulated before this match, the denominator of the *90 columns."
    )
    xg90: F[float] = pa.Field(description="prev_xg per 90 minutes.")
    xa90: F[float] = pa.Field(description="prev_xa per 90 minutes.")
    form90: F[float] = pa.Field(
        description="Combined form score per 90 minutes; formula in player_form."
    )

    # Pre-match fatigue (build_fatigue.player_fatigue); league matches only
    rest_days: F[int] = pa.Field(
        nullable=True, description="Days since the player's previous match. Null on a debut."
    )
    min_7d: F[int] = pa.Field(description="Minutes played in the 7 days before this match.")
    n_7d: F[int] = pa.Field(description="Matches played in the 7 days before this match.")
    min_28d: F[int] = pa.Field(description="Minutes played in the 28 days before this match.")
    streak_prev: F[int] = pa.Field(
        description="Length of the player's run of consecutive matches before this one."
    )
    fatigue_score: F[float] = pa.Field(
        description=(
            "0 to 100, weighting min_7d, rest_days, streak_prev and min_7d against min_28d;"
            " weights in player_fatigue."
        )
    )


class MatchCols(pa.DataFrameModel):
    """matches.csv: one row per match. Player slot columns: see slot_columns()."""

    # Match
    league: F[str] = pa.Field(description='Understat league: "ENG-Premier League".')
    season: F[str] = pa.Field(description='Season label: "2014/15".')
    match_date: F[datetime.date] = pa.Field(description="Match date.")
    home_team: F[str] = pa.Field(description="Home side, Understat name after rename_team_names.")
    away_team: F[str] = pa.Field(description="Away side, Understat name after rename_team_names.")
    season_start: F[pl.Int32] = pa.Field(description="Season start year: 2014.")
    game_id: F[int] = pa.Field(description="Understat match id; one row per game_id.")

    # Result (targets), Understat pitch score
    full_time_home_goals: F[int] = pa.Field(description="Home goals at full time.")
    full_time_away_goals: F[int] = pa.Field(description="Away goals at full time.")
    full_time_result: F[str] = pa.Field(
        isin=["H", "D", "A"], description="H home win, D draw, A away win."
    )
    home_xg: F[float] = pa.Field(description="Home expected goals.")
    away_xg: F[float] = pa.Field(description="Away expected goals.")

    # Post-match statistics summed from Understat player rows (_add_team_stats);
    # null for the few matches without player data. Not for pre-match features.
    home_shots: F[int] = pa.Field(nullable=True, description="Home shots.")
    away_shots: F[int] = pa.Field(nullable=True, description="Away shots.")
    home_yellow_cards: F[int] = pa.Field(nullable=True, description="Home yellow cards.")
    away_yellow_cards: F[int] = pa.Field(nullable=True, description="Away yellow cards.")
    home_red_cards: F[int] = pa.Field(nullable=True, description="Home red cards.")
    away_red_cards: F[int] = pa.Field(nullable=True, description="Away red cards.")

    # Pre-match Elo (_add_elo), per league from 2014/15; treat 2014/15 as warm-up
    elo_home: F[float] = pa.Field(description="Home side's Elo rating before kick-off.")
    elo_away: F[float] = pa.Field(description="Away side's Elo rating before kick-off.")
    elo_diff: F[float] = pa.Field(description="elo_home - elo_away.")

    # Player-based features below are null for the two matches Understat has no
    # line-ups for (2016/17 Bastia-Lyon, abandoned; one Bundesliga 2024/25 game).
    h_fatigue: F[float] = pa.Field(
        nullable=True, description="Home starters' fatigue_score combined by team_fatigue."
    )
    a_fatigue: F[float] = pa.Field(
        nullable=True, description="Away starters' fatigue_score combined by team_fatigue."
    )
    h_min_7d: F[int] = pa.Field(
        nullable=True, description="Home starters' min_7d combined by team_fatigue."
    )
    a_min_7d: F[int] = pa.Field(
        nullable=True, description="Away starters' min_7d combined by team_fatigue."
    )
    fatigue_diff: F[float] = pa.Field(nullable=True, description="h_fatigue - a_fatigue.")

    # Missing regulars (build_absence.team_absence)
    h_missing_q: F[float] = pa.Field(
        nullable=True,
        description="Quality of the home side's usual starters who do not start this match.",
    )
    a_missing_q: F[float] = pa.Field(
        nullable=True,
        description="Quality of the away side's usual starters who do not start this match.",
    )
    h_missing_regulars: F[int] = pa.Field(
        nullable=True, description="Home regulars (high EWMA start probability) not starting."
    )
    a_missing_regulars: F[int] = pa.Field(
        nullable=True, description="Away regulars (high EWMA start probability) not starting."
    )
    missing_q_diff: F[float] = pa.Field(nullable=True, description="h_missing_q - a_missing_q.")

    # Transfermarkt game info; null where no Transfermarkt game matched
    tm_game_id: F[str] = pa.Field(
        nullable=True, description="Matched Transfermarkt game id (std_tm.csv tm_game_id)."
    )
    tm_season: F[str] = pa.Field(nullable=True, description='Transfermarkt season: "2014".')
    tm_round: F[str] = pa.Field(nullable=True, description='Transfermarkt round: "1. Matchday".')
    h_formation: F[str] = pa.Field(
        nullable=True, description='Home starting formation: "4-2-3-1".'
    )
    a_formation: F[str] = pa.Field(nullable=True, description="Away starting formation.")

    result_overturned: F[bool] = pa.Field(
        description=(
            "True when the Understat pitch score differs from Transfermarkt's official "
            "result; False when they agree or no Transfermarkt game matched. Drop the True "
            "rows when training."
        )
    )


class CleanedMatchCols(ScheduleCols, FormationCols):
    """stat_matches.csv: xg.csv rows with their Transfermarkt league game (join_match_w_stat).

    Left join on league, season_start, home_team and away_team (club ids mapped to
    teams through clubs.csv), not on the date: postponed or resumed games can sit days
    apart in the two sources. Same rows as xg.csv, one per game_id; the Transfermarkt
    columns, the encoded formations of FormationCols included, are null where no game
    matched. Key: game_id.
    """

    tm_game_id: F[str] = pa.Field(
        nullable=True,
        description=(
            "Transfermarkt game id of the same fixture (std_tm.csv tm_game_id). Null when "
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


class ClubMapCols(pa.DataFrameModel):
    """clubs.csv: Transfermarkt club_id -> Understat team, from build_team_calendar.map_clubs.

    Learned from league fixtures (same league, score and date), not from names, so
    it holds even where the two sources spell a club differently. Key: club_id.
    """

    club_id: F[pl.Int32] = pa.Field(
        unique=True,
        description="Transfermarkt club id (home_club_id / away_club_id in std_tm.csv). Unique.",
    )

    team: F[str] = pa.Field(
        description="Understat team name the club pairs with most often, after rename_team_names."
    )

    n: F[pl.UInt32] = pa.Field(
        description=(
            "League fixtures that back the pairing: Transfermarkt games whose league, score"
            " and date (within max_day_shift days) match an Understat game with this team "
            "on the same side."
        )
    )

    games: F[pl.UInt32] = pa.Field(
        description="All the club's Transfermarkt league games, home and away."
    )

    coverage: F[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description=(
            "n / games, rounded to 3 places. Close to 1.0 when the pairing is right; "
            "_join_club_w_team rejects below 0.9 once games >= 10."
        ),
    )

    duplicate: F[bool] = pa.Field(
        description=(
            "True when another club_id pairs with the same team; _join_club_w_team rejects "
            "any True row."
        )
    )


PM = PlayerMatchCols
M = MatchCols
CLUB_MAP = ClubMapCols
CLEANED_MATCH = CleanedMatchCols


def slot_column(side: str, slot: int, feature: str) -> str:
    """Per-slot player column name: ("h", 1, "xg90") -> "h_player1_xg90"."""
    return f"{side}_player{slot}_{feature}"


def slot_columns() -> dict[str, pa.Column]:
    """The 88 generated player slot columns of matches.csv, in canonical order.

    One column per side (h, a), starter slot (1 to 11) and PLAYER_SLOT_FEATURES,
    holding that starter's pre-match value from player_matches.csv.
    """
    return {
        # nullable: no line-up data, or fewer than 11 starters recorded
        slot_column(side, slot, feat): pa.Column(
            pl.Float64,
            nullable=True,
            description=f"{feat} of {'home' if side == 'h' else 'away'} starter {slot}.",
        )
        for side in ("h", "a")
        for slot in range(1, N_SLOTS + 1)
        for feat in PLAYER_SLOT_FEATURES
    }


__all__ = [
    "CLUB_MAP",
    "M",
    "N_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PM",
    "CLEANED_MATCH",
    "ClubMapCols",
    "MatchCols",
    "PlayerMatchCols",
    "CleanedMatchCols",
    "slot_column",
    "slot_columns",
]
