"""Column names and dtypes of the clean outputs in settings.clean.out_dir.

The source frames (xg, player_stat) hold only the raw columns
the pipeline reads (schema.raw.SELECTED), renamed to snake_case.

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

PLAYER_SLOT_FEATURES = ("xg90", "xa90", "form90", "prev_min", "fatigue_score")
N_SLOTS = 11


@dataclass(frozen=True)
class ScheduleCols:
    """xg.csv: Understat schedule, played matches only, with goals and xG."""

    league: Column = Column("league", STR)
    season: Column = Column("season", INT)  # 1415
    game_id: Column = Column("game_id", INT)
    date: Column = Column("date", DATETIME)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    home_goals: Column = Column("home_goals", INT, nullable=True)  # null before kick-off
    away_goals: Column = Column("away_goals", INT, nullable=True)
    home_xg: Column = Column("home_xg", FLOAT, nullable=True)
    away_xg: Column = Column("away_xg", FLOAT, nullable=True)
    is_result: Column = Column("is_result", BOOL)
    season_start: Column = Column("season_start", INT)


@dataclass(frozen=True)
class PlayerStatCols:
    """player_stat.csv: Understat player match stats, one row per player per match."""

    league: Column = Column("league", STR)
    season: Column = Column("season", STR)  # "1415"; CSV reads it back as an integer
    game: Column = Column("game", STR)
    team: Column = Column("team", STR)
    player: Column = Column("player", STR)
    game_id: Column = Column("game_id", INT)
    team_id: Column = Column("team_id", INT)
    player_id: Column = Column("player_id", INT)
    position: Column = Column("position", STR)  # "Sub" for substitutes
    minutes: Column = Column("minutes", INT)
    xg: Column = Column("xg", FLOAT)
    xa: Column = Column("xa", FLOAT)
    goals: Column = Column("goals", INT)
    own_goals: Column = Column("own_goals", INT)
    shots: Column = Column("shots", INT)
    xg_chain: Column = Column("xg_chain", FLOAT)
    xg_buildup: Column = Column("xg_buildup", FLOAT)
    assists: Column = Column("assists", INT)
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
    home_xg: Column = Column("home_xg", FLOAT)
    away_xg: Column = Column("away_xg", FLOAT)
    # Post-match statistics summed from Understat player rows (_add_team_stats);
    # null for the few matches without player data.
    home_shots: Column = Column("home_shots", INT, nullable=True)
    away_shots: Column = Column("away_shots", INT, nullable=True)
    home_yellow_cards: Column = Column("home_yellow_cards", INT, nullable=True)
    away_yellow_cards: Column = Column("away_yellow_cards", INT, nullable=True)
    home_red_cards: Column = Column("home_red_cards", INT, nullable=True)
    away_red_cards: Column = Column("away_red_cards", INT, nullable=True)
    # Pre-match features
    elo_home: Column = Column("elo_home", FLOAT)
    elo_away: Column = Column("elo_away", FLOAT)
    elo_diff: Column = Column("elo_diff", FLOAT)
    # Player-based features below are null for the two matches Understat has no
    # line-ups for (2016/17 Bastia-Lyon, abandoned; one Bundesliga 2024/25 game).
    h_fatigue: Column = Column("h_fatigue", FLOAT, nullable=True)
    a_fatigue: Column = Column("a_fatigue", FLOAT, nullable=True)
    h_min_7d: Column = Column("h_min_7d", INT, nullable=True)
    a_min_7d: Column = Column("a_min_7d", INT, nullable=True)
    fatigue_diff: Column = Column("fatigue_diff", FLOAT, nullable=True)

    # Missing regulars (build_absence.team_absence)
    h_missing_q: Column = Column("h_missing_q", FLOAT, nullable=True)
    a_missing_q: Column = Column("a_missing_q", FLOAT, nullable=True)
    h_missing_regulars: Column = Column("h_missing_regulars", INT, nullable=True)
    a_missing_regulars: Column = Column("a_missing_regulars", INT, nullable=True)
    missing_q_diff: Column = Column("missing_q_diff", FLOAT, nullable=True)

    # Transfermarkt game info (build_team_calendar.game_info); null where no TM game matched
    tm_game_id: Column = Column("tm_game_id", STR, nullable=True)
    tm_season: Column = Column("tm_season", STR, nullable=True)
    tm_round: Column = Column("tm_round", STR, nullable=True)
    h_formation: Column = Column("h_formation", STR, nullable=True)
    a_formation: Column = Column("a_formation", STR, nullable=True)
    # Pitch score differs from Transfermarkt's official one: drop these when training
    result_overturned: Column = Column("result_overturned", BOOL)


SCHEDULE = ScheduleCols()
PLAYER_STAT = PlayerStatCols()
PM = PlayerMatchCols()
M = MatchCols()


def slot_column(side: str, slot: int, feature: str) -> str:
    """Per-slot player column name: ("h", 1, "xg90") -> "h_player1_xg90"."""
    return f"{side}_player{slot}_{feature}"


def slot_columns() -> list[Column]:
    """The 88 generated player slot columns of matches.csv."""
    return [
        # nullable: no line-up data, or fewer than 11 starters recorded
        Column(slot_column(side, slot, feat), FLOAT, nullable=True)
        for side in ("h", "a")
        for slot in range(1, N_SLOTS + 1)
        for feat in PLAYER_SLOT_FEATURES
    ]


# File name (without .csv) -> every column expected in that file.
CLEAN_FILES: dict[str, list[Column]] = {
    "xg": columns(SCHEDULE),
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
