"""Pre-match features per match, and the match table with all of them joined on.

    MatchSideCols      date and home side, what get_fatigue reads from a match
    EloInputCols       get_elo's input; EloCols its output; PredictionCols walk_forward's
    TeamFatigueCols    fatigue.csv: starters' fatigue per side
    TeamScheduleCols   schedule.csv: each side's full-calendar schedule features
    PLAYER_SLOTS       players.csv: each side's starters as slot columns
    MatchFeaturesCols  match_features.csv: cleaned_matches plus every feature above
    MissingCols        missing.csv: every null cell of match_features, with its reason

Every feature uses only what was known at kick-off.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData
from pandera.typing import FieldType as F

from money_printer_research.schema.match.clean import CleanedMatchCols


class MatchSideCols(pa.DataFrameModel):
    """get_fatigue's view of a match: when it was played and who was at home."""

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    match_date: F[datetime.date] = pa.Field(description="Date part of the kick-off.")
    home_team: F[str] = pa.Field(description="Home side, after norm_team_names.")


OUTCOMES = ("H", "D", "A")


class EloInputCols(pa.DataFrameModel):
    """What get_elo reads: one row per played match, built by to_elo_input."""

    season: F[str] = pa.Field(
        description='Season label: "2014/15". A change of season triggers promotion and regress.'
    )
    match_date: F[datetime.date] = pa.Field(
        description="Match date; get_elo replays in this order."
    )
    home_team: F[str] = pa.Field(description="Home side, Understat name after norm_team_names.")
    away_team: F[str] = pa.Field(description="Away side, Understat name after norm_team_names.")
    full_time_home_goals: F[int] = pa.Field(ge=0, description="Home goals at full time.")
    full_time_away_goals: F[int] = pa.Field(ge=0, description="Away goals at full time.")
    full_time_result: F[str] = pa.Field(
        isin=list(OUTCOMES), description="H home win, D draw, A away win; the target of evaluate."
    )


class EloCols(EloInputCols):
    """get_elo's output: the input plus pre-match ratings."""

    elo_home: F[float] = pa.Field(description="Home side's Elo rating before kick-off.")
    elo_away: F[float] = pa.Field(description="Away side's Elo rating before kick-off.")
    elo_diff: F[float] = pa.Field(
        description="elo_home - elo_away, without home_adv; the model's only regressor."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def diff_is_home_minus_away(self, data: PolarsData) -> pl.LazyFrame:
        """elo_diff equals elo_home - elo_away."""
        return data.lazyframe.select(
            ((pl.col("elo_home") - pl.col("elo_away") - pl.col("elo_diff")).abs() < 1e-9).alias(
                "elo_diff"
            )
        )


class PredictionCols(EloCols):
    """walk_forward's output: the test seasons with expected goals and H/D/A probabilities."""

    lam_home: F[float] = pa.Field(gt=0, description="Expected home goals (Poisson rate).")
    lam_away: F[float] = pa.Field(gt=0, description="Expected away goals (Poisson rate).")
    p_home: F[float] = pa.Field(ge=0, le=1, description="P(home win) from the score grid.")
    p_draw: F[float] = pa.Field(ge=0, le=1, description="P(draw) from the score grid.")
    p_away: F[float] = pa.Field(ge=0, le=1, description="P(away win) from the score grid.")
    rho: F[float] = pa.Field(
        description="Dixon-Coles rho fitted on the training seasons; 0.0 without Dixon-Coles."
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def probabilities_sum_to_one(self, data: PolarsData) -> pl.LazyFrame:
        """p_home + p_draw + p_away is 1 for every match."""
        total = pl.col("p_home") + pl.col("p_draw") + pl.col("p_away")
        return data.lazyframe.select(((total - 1).abs() < 1e-9).alias("p_sum"))


class TeamFatigueCols(pa.DataFrameModel):
    """team_fatigue's output: starters' fatigue per side, one row per game_id."""

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    h_fatigue: F[float] = pa.Field(
        nullable=True, description="Mean fatigue_score of the home starters."
    )
    a_fatigue: F[float] = pa.Field(
        nullable=True, description="Mean fatigue_score of the away starters."
    )
    h_min_7d: F[int] = pa.Field(
        nullable=True, description="Sum of the home starters' min_7d. Null without home starters."
    )
    a_min_7d: F[int] = pa.Field(
        nullable=True, description="Sum of the away starters' min_7d. Null without away starters."
    )
    fatigue_diff: F[float] = pa.Field(nullable=True, description="h_fatigue - a_fatigue.")


class TeamScheduleCols(pa.DataFrameModel):
    """team_schedule_features: per Understat game_id, each side's ClubCalendarCols features.

    Null on a side whose club has no matched Transfermarkt game.
    """

    game_id: F[int] = pa.Field(unique=True, description="Understat match id.")
    h_rest_days_all: F[int] = pa.Field(nullable=True, description="Home rest_days_all.")
    h_games_7d_all: F[int] = pa.Field(nullable=True, description="Home games_7d_all.")
    h_games_14d_all: F[int] = pa.Field(nullable=True, description="Home games_14d_all.")
    h_non_league_14d: F[int] = pa.Field(nullable=True, description="Home non_league_14d.")
    h_prev_europe_away: F[bool] = pa.Field(nullable=True, description="Home prev_europe_away.")
    h_days_to_next_all: F[int] = pa.Field(nullable=True, description="Home days_to_next_all.")
    h_next_is_europe: F[bool] = pa.Field(nullable=True, description="Home next_is_europe.")
    a_rest_days_all: F[int] = pa.Field(nullable=True, description="Away rest_days_all.")
    a_games_7d_all: F[int] = pa.Field(nullable=True, description="Away games_7d_all.")
    a_games_14d_all: F[int] = pa.Field(nullable=True, description="Away games_14d_all.")
    a_non_league_14d: F[int] = pa.Field(nullable=True, description="Away non_league_14d.")
    a_prev_europe_away: F[bool] = pa.Field(nullable=True, description="Away prev_europe_away.")
    a_days_to_next_all: F[int] = pa.Field(nullable=True, description="Away days_to_next_all.")
    a_next_is_europe: F[bool] = pa.Field(nullable=True, description="Away next_is_europe.")


PLAYER_SLOT_FEATURES = ("xg90", "xa90", "form90", "prev_min", "fatigue_score")


N_SLOTS = 11


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


class MatchFeaturesCols(CleanedMatchCols, TeamFatigueCols, TeamScheduleCols):
    """match_features.csv: cleaned_matches with every pre-match feature joined on (join_features).

    One row per Understat game_id, the rows of cleaned_matches. CleanedMatchCols gives the
    match, its result and the Transfermarkt columns; TeamFatigueCols the h_/a_ fatigue;
    TeamScheduleCols the h_/a_ full-calendar schedule; the Elo columns are below. A feature
    is null where its source has no value for the game; missing.csv (MissingCols) says why,
    cell by cell. Key: game_id.
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


MISSING_REASONS = (
    "no_lineup",
    "short_lineup",
    "too_few_minutes",
    "no_schedule",
    "no_next_game",
    "no_tm_game",
    "other",
)
"""Why a cell of match_features is null; see features.missing for each one."""


class MissingCols(pa.DataFrameModel):
    """missing.csv: one row per null cell of match_features, and why it is null.

    Built by features.missing.missing_table. Key: game_id and column.
    """

    game_id: F[int] = pa.Field(description="Understat match id: the row of match_features.")
    column: F[str] = pa.Field(description='The null column of match_features: "h_player3_xg90".')
    side: F[str] = pa.Field(
        nullable=True,
        isin=["h", "a"],
        description="h or a for a column of one side; null for one of the match (fatigue_diff).",
    )
    reason: F[str] = pa.Field(
        isin=list(MISSING_REASONS),
        description=(
            "no_lineup: Understat has no starters for the side. short_lineup: fewer than 11"
            " starters, so this slot is empty. too_few_minutes: the starter has under"
            " MIN_PREV_MIN previous minutes. no_schedule: the club has no Transfermarkt game."
            " no_next_game: the club's last game in the data. no_tm_game: the match has no"
            " Transfermarkt game. other: none of these."
        ),
    )

    @pa.dataframe_check  # pyright: ignore[reportUnknownMemberType]  (pandera is untyped here)
    def one_row_per_cell(self, data: PolarsData) -> pl.LazyFrame:
        """game_id and column together are unique."""
        return data.lazyframe.select(
            pl.struct("game_id", "column").is_unique().alias("one_row_per_cell")
        )


def match_features_schema() -> pa.DataFrameSchema:
    """MatchFeaturesCols plus the player slot columns: the whole of match_features.csv."""
    return MatchFeaturesCols.to_schema().add_columns(slot_columns())


__all__ = [
    "EloCols",
    "EloInputCols",
    "MISSING_REASONS",
    "MatchFeaturesCols",
    "MatchSideCols",
    "MissingCols",
    "N_SLOTS",
    "OUTCOMES",
    "PLAYER_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PredictionCols",
    "TeamFatigueCols",
    "TeamScheduleCols",
    "match_features_schema",
    "slot_column",
    "slot_columns",
]
