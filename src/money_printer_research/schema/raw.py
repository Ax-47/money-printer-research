"""Column names and dtypes of the raw source files, as frozen dataclass fields.

Each field is a Column: a str (the raw column name) that also carries its
expected polars dtype, so it works anywhere a column name does:

    pl.col(KG.home_team)            # "HomeTeam"
    validate(df, KG)                # missing columns and wrong dtypes, in one error
    pl.read_csv(path, schema_overrides=dtypes(XG))

Raw names are kept as they come from each source (Kaggle uses CamelCase);
snake_columns_pl turns them into the clean names used in schema.py.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from money_printer_research.config import Settings, settings
from money_printer_research.schema.column import (
    BOOL,
    FLOAT,
    INT,
    STR,
    Column,
    columns,
    dtypes,
    validate,
)


@dataclass(frozen=True)
class KaggleRaw:
    """data/epl_final.csv (Kaggle, Premier League 2000/01 onwards). Read with try_parse_dates."""

    season: Column = Column("Season", STR)  # "2000/01"
    match_date: Column = Column("MatchDate", pl.Date())
    home_team: Column = Column("HomeTeam", STR)
    away_team: Column = Column("AwayTeam", STR)
    full_time_home_goals: Column = Column("FullTimeHomeGoals", INT)
    full_time_away_goals: Column = Column("FullTimeAwayGoals", INT)
    full_time_result: Column = Column("FullTimeResult", STR)  # "H" / "D" / "A"
    half_time_home_goals: Column = Column("HalfTimeHomeGoals", INT)
    half_time_away_goals: Column = Column("HalfTimeAwayGoals", INT)
    half_time_result: Column = Column("HalfTimeResult", STR)
    home_shots: Column = Column("HomeShots", INT)
    away_shots: Column = Column("AwayShots", INT)
    home_shots_on_target: Column = Column("HomeShotsOnTarget", INT)
    away_shots_on_target: Column = Column("AwayShotsOnTarget", INT)
    home_corners: Column = Column("HomeCorners", INT)
    away_corners: Column = Column("AwayCorners", INT)
    home_fouls: Column = Column("HomeFouls", INT)
    away_fouls: Column = Column("AwayFouls", INT)
    home_yellow_cards: Column = Column("HomeYellowCards", INT)
    away_yellow_cards: Column = Column("AwayYellowCards", INT)
    home_red_cards: Column = Column("HomeRedCards", INT)
    away_red_cards: Column = Column("AwayRedCards", INT)


@dataclass(frozen=True)
class UnderstatScheduleRaw:
    """data/raw/xg_schedule.csv (Understat read_schedule). Read with try_parse_dates."""

    league: Column = Column("league", STR)  # "ENG-Premier League"
    season: Column = Column("season", INT)  # 1415
    game: Column = Column("game", STR)  # "2014-08-16 Arsenal-Crystal Palace"
    league_id: Column = Column("league_id", INT)
    season_id: Column = Column("season_id", INT)
    game_id: Column = Column("game_id", INT)
    date: Column = Column("date", pl.Datetime("us"))
    home_team_id: Column = Column("home_team_id", INT)
    away_team_id: Column = Column("away_team_id", INT)
    home_team: Column = Column("home_team", STR)
    away_team: Column = Column("away_team", STR)
    away_team_code: Column = Column("away_team_code", STR)
    home_team_code: Column = Column("home_team_code", STR)
    home_goals: Column = Column("home_goals", INT)
    away_goals: Column = Column("away_goals", INT)
    home_xg: Column = Column("home_xg", FLOAT)
    away_xg: Column = Column("away_xg", FLOAT)
    is_result: Column = Column("is_result", BOOL)  # False for fixtures not yet played
    has_data: Column = Column("has_data", BOOL)
    url: Column = Column("url", STR)


@dataclass(frozen=True)
class UnderstatPlayerRaw:
    """data/raw/player_stats/<league>/<season>.parquet (Understat read_player_match_stats)."""

    league: Column = Column("league", STR)
    season: Column = Column("season", STR)  # "1415", a string here unlike the schedule
    game: Column = Column("game", STR)
    team: Column = Column("team", STR)
    player: Column = Column("player", STR)
    league_id: Column = Column("league_id", STR)  # "1"; a string in this file only
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


@dataclass(frozen=True)
class FBrefLineupRaw:
    """data/raw/lineups/<season>.parquet (FBref read_lineup)."""

    league: Column = Column("league", STR)
    season: Column = Column("season", STR)  # "2425"
    game: Column = Column("game", STR)  # "2024-08-16 Manchester Utd-Fulham"
    jersey_number: Column = Column("jersey_number", INT)
    player: Column = Column("player", STR)
    team: Column = Column("team", STR)
    is_starter: Column = Column("is_starter", BOOL)
    position: Column = Column("position", STR, nullable=True)  # null for substitutes
    minutes_played: Column = Column("minutes_played", INT)


KG = KaggleRaw()
XG = UnderstatScheduleRaw()
PS = UnderstatPlayerRaw()
FB = FBrefLineupRaw()

type RawSchema = KaggleRaw | UnderstatScheduleRaw | UnderstatPlayerRaw | FBrefLineupRaw


def raw_files(cfg: Settings) -> list[tuple[Path, RawSchema]]:
    """Every raw file the pipeline reads, paired with its schema."""
    files: list[tuple[Path, RawSchema]] = [
        (cfg.kaggle.output_dir / "epl_final.csv", KG),
        (cfg.understat.schedule_file, XG),
    ]
    files += [(p, PS) for p in sorted(cfg.understat.out_dir.glob("*/*.parquet"))]
    files += [(p, FB) for p in sorted(cfg.fbref.out_dir.glob("**/*.parquet"))]
    return files


def _read(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=True)


def main() -> None:
    """Entry point `check_raw`: validate every raw file; exit 1 if any fails."""
    failed, ok = 0, 0
    for path, schema in raw_files(settings):
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            failed += 1
            continue
        try:
            validate(_read(path), schema)
            ok += 1
        except ValueError as e:
            print(f"FAIL    {path}: {e}", flush=True)
            failed += 1
    print(f"{ok} ok, {failed} failed", flush=True)
    raise SystemExit(1 if failed else 0)


__all__ = ["FB", "KG", "PS", "XG", "Column", "columns", "dtypes", "main", "raw_files", "validate"]
