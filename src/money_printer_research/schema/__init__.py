"""Column names and dtypes of the clean outputs in settings.clean.out_dir.

Every field is a Column (see column.py): a str with the expected dtype attached.
The dtypes are the ones the pipeline produces in memory. CSV loses some of
them (season "1516" comes back as an integer), so read the CSVs with
read_clean(), which applies the schema, then validate().
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from money_printer_research.config import settings
from money_printer_research.schema.column import (
    BOOL,
    DATE,
    DATETIME,
    FLOAT,
    INT,
    STR,
    Column,
    check,
    columns,
    dtypes,
    require,
    validate,
)

PLAYER_SLOT_FEATURES = ("xg90", "xa90", "form90", "prev_min")
N_SLOTS = 11


@dataclass(frozen=True)
class KaggleCols:
    """kaggle.csv: Kaggle EPL matches after snake_case and team-name cleaning."""

    season: Column = Column("season", STR)  # "2000/01"
    match_date: Column = Column("match_date", DATE)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    full_time_home_goals: Column = Column("full_time_home_goals", INT)
    full_time_away_goals: Column = Column("full_time_away_goals", INT)
    full_time_result: Column = Column("full_time_result", STR)
    half_time_home_goals: Column = Column("half_time_home_goals", INT)
    half_time_away_goals: Column = Column("half_time_away_goals", INT)
    half_time_result: Column = Column("half_time_result", STR)
    home_shots: Column = Column("home_shots", INT)
    away_shots: Column = Column("away_shots", INT)
    home_shots_on_target: Column = Column("home_shots_on_target", INT)
    away_shots_on_target: Column = Column("away_shots_on_target", INT)
    home_corners: Column = Column("home_corners", INT)
    away_corners: Column = Column("away_corners", INT)
    home_fouls: Column = Column("home_fouls", INT)
    away_fouls: Column = Column("away_fouls", INT)
    home_yellow_cards: Column = Column("home_yellow_cards", INT)
    away_yellow_cards: Column = Column("away_yellow_cards", INT)
    home_red_cards: Column = Column("home_red_cards", INT)
    away_red_cards: Column = Column("away_red_cards", INT)
    season_start: Column = Column("season_start", INT)


@dataclass(frozen=True)
class ScheduleCols:
    """xg.csv: Understat schedule with goals and xG."""

    league: Column = Column("league", STR)
    season: Column = Column("season", INT)  # 1415
    game: Column = Column("game", STR)
    league_id: Column = Column("league_id", INT)
    season_id: Column = Column("season_id", INT)
    game_id: Column = Column("game_id", INT)
    date: Column = Column("date", DATETIME)
    home_team_id: Column = Column("home_team_id", INT)
    away_team_id: Column = Column("away_team_id", INT)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    away_team_code: Column = Column("away_team_code", STR)
    home_team_code: Column = Column("home_team_code", STR)
    home_goals: Column = Column("home_goals", INT, nullable=True)  # null before kick-off
    away_goals: Column = Column("away_goals", INT, nullable=True)
    home_xg: Column = Column("home_xg", FLOAT, nullable=True)
    away_xg: Column = Column("away_xg", FLOAT, nullable=True)
    is_result: Column = Column("is_result", BOOL)
    has_data: Column = Column("has_data", BOOL)
    url: Column = Column("url", STR)
    season_start: Column = Column("season_start", INT)


@dataclass(frozen=True)
class FbrefCols:
    """fbref.csv: FBref line-ups, one row per player per match."""

    league: Column = Column("league", STR)
    season: Column = Column("season", STR)  # "1516"; CSV reads it back as an integer
    game: Column = Column("game", STR)
    jersey_number: Column = Column("jersey_number", INT)
    player: Column = Column("player", STR)
    team: Column = Column("team", STR)
    is_starter: Column = Column("is_starter", BOOL)
    position: Column = Column("position", STR, nullable=True)  # null for substitutes
    minutes_played: Column = Column("minutes_played", INT)
    season_start: Column = Column("season_start", INT)


@dataclass(frozen=True)
class PlayerStatCols:
    """player_stat.csv: Understat player match stats, one row per player per match."""

    league: Column = Column("league", STR)
    season: Column = Column("season", STR)  # "1415"; CSV reads it back as an integer
    game: Column = Column("game", STR)
    team: Column = Column("team", STR)
    player: Column = Column("player", STR)
    league_id: Column = Column("league_id", STR)  # "1"; CSV reads it back as an integer
    season_id: Column = Column("season_id", INT)
    game_id: Column = Column("game_id", INT)
    team_id: Column = Column("team_id", INT)
    player_id: Column = Column("player_id", INT)
    position: Column = Column("position", STR)  # "Sub" for substitutes
    position_id: Column = Column("position_id", INT)
    minutes: Column = Column("minutes", INT)
    goals: Column = Column("goals", INT)
    own_goals: Column = Column("own_goals", INT)
    shots: Column = Column("shots", INT)
    xg: Column = Column("xg", FLOAT)
    xg_chain: Column = Column("xg_chain", FLOAT)
    xg_buildup: Column = Column("xg_buildup", FLOAT)
    assists: Column = Column("assists", INT)
    xa: Column = Column("xa", FLOAT)
    key_passes: Column = Column("key_passes", INT)
    yellow_cards: Column = Column("yellow_cards", INT)
    red_cards: Column = Column("red_cards", INT)
    season_start: Column = Column("season_start", INT)


@dataclass(frozen=True)
class PlayerMatchCols(PlayerStatCols):
    """player_matches.csv: player_stat plus match context, pre-match form and fatigue."""

    # Match context (_build_player_matches)
    match_date: Column = Column("match_date", DATE)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    is_home: Column = Column("is_home", BOOL)
    opponent: Column = Column("opponent", STR)
    # Pre-match form (build_player.player_form)
    prev_xg: Column = Column("prev_xg", FLOAT)
    prev_xa: Column = Column("prev_xa", FLOAT)
    prev_min: Column = Column("prev_min", INT)
    xg90: Column = Column("xg90", FLOAT)
    xa90: Column = Column("xa90", FLOAT)
    form90: Column = Column("form90", FLOAT)
    # Pre-match fatigue (build_fatigue.player_fatigue)
    rest_days: Column = Column("rest_days", INT, nullable=True)  # null on a debut
    min_7d: Column = Column("min_7d", INT)
    n_7d: Column = Column("n_7d", INT)
    min_28d: Column = Column("min_28d", INT)
    streak_prev: Column = Column("streak_prev", INT)
    fatigue_score: Column = Column("fatigue_score", FLOAT)


@dataclass(frozen=True)
class MatchCols:
    """matches.csv: one row per match. Player slot columns: see slot_columns()."""

    # Match
    league: Column = Column("league", STR, required=False)  # multi-league pipeline only
    season: Column = Column("season", STR)  # "2014/15"
    match_date: Column = Column("match_date", DATE)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    season_start: Column = Column("season_start", INT)
    game_id: Column = Column("game_id", INT)
    # Result (targets)
    full_time_home_goals: Column = Column("full_time_home_goals", INT)
    full_time_away_goals: Column = Column("full_time_away_goals", INT)
    full_time_result: Column = Column("full_time_result", STR)
    # Post-match statistics. Kaggle covers the Premier League only, so these are
    # null for other leagues and for the matches Kaggle is missing.
    half_time_home_goals: Column = Column("half_time_home_goals", INT, nullable=True)
    half_time_away_goals: Column = Column("half_time_away_goals", INT, nullable=True)
    half_time_result: Column = Column("half_time_result", STR, nullable=True)
    home_shots: Column = Column("home_shots", INT, nullable=True)
    away_shots: Column = Column("away_shots", INT, nullable=True)
    home_shots_on_target: Column = Column("home_shots_on_target", INT, nullable=True)
    away_shots_on_target: Column = Column("away_shots_on_target", INT, nullable=True)
    home_corners: Column = Column("home_corners", INT, nullable=True)
    away_corners: Column = Column("away_corners", INT, nullable=True)
    home_fouls: Column = Column("home_fouls", INT, nullable=True)
    away_fouls: Column = Column("away_fouls", INT, nullable=True)
    home_yellow_cards: Column = Column("home_yellow_cards", INT, nullable=True)
    away_yellow_cards: Column = Column("away_yellow_cards", INT, nullable=True)
    home_red_cards: Column = Column("home_red_cards", INT, nullable=True)
    away_red_cards: Column = Column("away_red_cards", INT, nullable=True)
    home_xg: Column = Column("home_xg", FLOAT)
    away_xg: Column = Column("away_xg", FLOAT)
    # Pre-match features
    elo_home: Column = Column("elo_home", FLOAT)
    elo_away: Column = Column("elo_away", FLOAT)
    elo_diff: Column = Column("elo_diff", FLOAT)
    h_fatigue: Column = Column("h_fatigue", FLOAT)
    a_fatigue: Column = Column("a_fatigue", FLOAT)
    h_min_7d: Column = Column("h_min_7d", INT)
    a_min_7d: Column = Column("a_min_7d", INT)
    fatigue_diff: Column = Column("fatigue_diff", FLOAT)


KAGGLE = KaggleCols()
SCHEDULE = ScheduleCols()
FBREF = FbrefCols()
PLAYER_STAT = PlayerStatCols()
PM = PlayerMatchCols()
M = MatchCols()


def slot_column(side: str, slot: int, feature: str) -> str:
    """Per-slot player column name: ("h", 1, "xg90") -> "h_player1_xg90"."""
    return f"{side}_player{slot}_{feature}"


def slot_columns() -> list[Column]:
    """The 88 generated player slot columns of matches.csv."""
    return [
        Column(slot_column(side, slot, feat), FLOAT)
        for side in ("h", "a")
        for slot in range(1, N_SLOTS + 1)
        for feat in PLAYER_SLOT_FEATURES
    ]


# File name (without .csv) -> every column expected in that file.
CLEAN_FILES: dict[str, list[Column]] = {
    "kaggle": columns(KAGGLE),
    "xg": columns(SCHEDULE),
    "fbref": columns(FBREF),
    "player_stat": columns(PLAYER_STAT),
    "player_matches": columns(PM),
    "matches": columns(M) + slot_columns(),
}


def read_clean(out_dir: Path, name: str) -> pl.DataFrame:
    """Read <out_dir>/<name>.csv with the schema's dtypes, then check it."""
    cols = CLEAN_FILES[name]
    df = pl.read_csv(out_dir / f"{name}.csv", schema_overrides=dtypes(*cols))
    check(df, cols, name)
    return df


def main() -> None:
    """Entry point `check_clean`: validate every clean CSV; exit 1 if any fails."""
    out = settings.clean.out_dir
    failed = 0
    for name in CLEAN_FILES:
        path = out / f"{name}.csv"
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            failed += 1
            continue
        try:
            df = read_clean(out, name)
            print(f"ok      {name:15s} {df.height:>8,} rows", flush=True)
        except ValueError as e:
            print(f"FAIL    {e}", flush=True)
            failed += 1
    raise SystemExit(1 if failed else 0)


__all__ = [
    "FBREF",
    "KAGGLE",
    "M",
    "PLAYER_STAT",
    "PM",
    "SCHEDULE",
    "Column",
    "columns",
    "main",
    "read_clean",
    "require",
    "slot_column",
    "slot_columns",
    "validate",
]
