"""Joined outputs: frames built from more than one clean frame.

    clubs.csv            ClubMapCols
    cleaned_matches.csv  CleanedMatchCols
    match_features.csv   MatchFeaturesCols: CleanedMatchCols plus Elo, fatigue and schedule,
                         plus the player slot columns (match_features_schema() checks all)
    players.csv          PLAYER_SLOTS: game_id plus slot_columns(), get_player_features' output

The single-source frames they are built from are in schema/clean.py; the fatigue and
schedule feature frames in schema/fatigue.py and the calendar schema module.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

from money_printer_research.schema.calendar import TeamScheduleCols
from money_printer_research.schema.clean import ScheduleCols
from money_printer_research.schema.fatigue import TeamFatigueCols
from money_printer_research.schema.formation import FormationCols

PLAYER_SLOT_FEATURES = ("xg90", "xa90", "form90", "prev_min", "fatigue_score")
N_SLOTS = 11


class CleanedMatchCols(ScheduleCols, FormationCols):
    """cleaned_matches.csv: each Understat match with its Transfermarkt league game.

    Built by join_match_w_stat from the cleaned schedule (ScheduleCols).

    Left join on league, season_start, home_team and away_team (club ids mapped to
    teams through clubs.csv), not on the date: postponed or resumed games can sit days
    apart in the two sources. Same rows as the schedule, one per game_id; the Transfermarkt
    columns, the encoded formations of FormationCols included, are null where no game
    matched. Key: game_id.
    """

    tm_game_id: F[str] = pa.Field(
        nullable=True,
        description=(
            "Transfermarkt game id of the same fixture (match_leagues.csv tm_game_id). Null when "
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


class MatchFeaturesCols(CleanedMatchCols, TeamFatigueCols, TeamScheduleCols):
    """match_features.csv: cleaned_matches with every pre-match feature joined on (join_features).

    One row per Understat game_id, the rows of cleaned_matches. CleanedMatchCols gives the
    match, its result and the Transfermarkt columns; TeamFatigueCols the h_/a_ fatigue;
    TeamScheduleCols the h_/a_ full-calendar schedule; the Elo columns are below. A feature
    is null where its source has no value for the game. Key: game_id.
    """

    game_id: F[int] = pa.Field(unique=True, description="Understat match id; one row per game.")

    match_date: F[datetime.date] = pa.Field(
        description="Date part of date; the key, with both teams, that Elo is joined on."
    )

    full_time_result: F[str] = pa.Field(
        isin=["H", "D", "A"],
        description="H home win, D draw, A away win, from home_goals and away_goals. The target.",
    )

    elo_home: F[float] = pa.Field(description="Home side's Elo rating before kick-off.")

    elo_away: F[float] = pa.Field(description="Away side's Elo rating before kick-off.")

    elo_diff: F[float] = pa.Field(
        description=(
            "elo_home - elo_away, without home advantage. Every team starts level in 2014/15,"
            " so treat that season as warm-up."
        )
    )


class ClubMapCols(pa.DataFrameModel):
    """clubs.csv: Transfermarkt club_id -> Understat team, from map_club_w_team.

    Learned from league fixtures (same league, score and date), not from names, so
    it holds even where the two sources spell a club differently. Key: club_id.
    """

    club_id: F[pl.Int32] = pa.Field(
        unique=True,
        description=(
            "Transfermarkt club id (home_club_id / away_club_id in match_leagues.csv). Unique."
        ),
    )

    team: F[str] = pa.Field(
        description="Understat team name the club pairs with most often, after norm_team_names."
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
            "map_club_w_team rejects below 0.9 once games >= 10."
        ),
    )

    duplicate: F[bool] = pa.Field(
        description=(
            "True when another club_id pairs with the same team; map_club_w_team rejects "
            "any True row."
        )
    )


CLUB_MAP = ClubMapCols
CLEANED_MATCH = CleanedMatchCols
MATCH_FEATURES = MatchFeaturesCols


def slot_column(side: str, slot: int, feature: str) -> str:
    """Per-slot player column name: ("h", 1, "xg90") -> "h_player1_xg90"."""
    return f"{side}_player{slot}_{feature}"


def slot_columns() -> dict[str, pa.Column]:
    """The 110 generated player slot columns of players.csv and match_features.csv, in order.

    One column per side (h, a), starter slot (1 to 11) and PLAYER_SLOT_FEATURES,
    holding that starter's pre-match value from player_form and player_fatigue.
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


PLAYER_SLOTS = pa.DataFrameSchema(
    {"game_id": pa.Column(int, unique=True), **slot_columns()},
    name="PlayerSlots",
    description="get_player_features: one row per game_id, each side's starters as slot columns.",
)


def match_features_schema() -> pa.DataFrameSchema:
    """MatchFeaturesCols plus the player slot columns: the whole of match_features.csv."""
    return MatchFeaturesCols.to_schema().add_columns(slot_columns())


__all__ = [
    "CLEANED_MATCH",
    "CLUB_MAP",
    "MATCH_FEATURES",
    "N_SLOTS",
    "PLAYER_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "CleanedMatchCols",
    "ClubMapCols",
    "MatchFeaturesCols",
    "match_features_schema",
    "slot_column",
    "slot_columns",
]
