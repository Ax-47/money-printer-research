"""Column names and dtypes of the raw source files, as frozen dataclass fields.

Each field is a Column: a str (the raw column name) that also carries its
expected polars dtype, so it works anywhere a column name does:

    pl.col(KG.home_team)            # "HomeTeam"
    validate(df, KG)                # missing columns and wrong dtypes, in one error
    pl.read_csv(path, schema_overrides=dtypes(XG))

Raw names are kept as they come from each source (Kaggle uses CamelCase);
snake_columns_pl turns them into the clean names used in schema.py.

Two uses, two column sets:
- collection checks every column of the raw schema (check_collected), so a
  source that changes shape fails at download time;
- the clean pipeline reads only the columns it uses (read_selected, SELECTED).
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from money_printer_research.config import Settings, settings
from money_printer_research.schema.column import (
    BOOL,
    DATE,
    FLOAT,
    INT,
    STR,
    Column,
    check,
    columns,
    dtypes,
    validate,
)

INT32 = pl.Int32()


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
    home_goals: Column = Column("home_goals", INT, nullable=True)  # null before kick-off
    away_goals: Column = Column("away_goals", INT, nullable=True)
    home_xg: Column = Column("home_xg", FLOAT, nullable=True)
    away_xg: Column = Column("away_xg", FLOAT, nullable=True)
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
class TransfermarktGamesRaw:
    """<transfermarkt.out_dir>/games.parquet (export of the Transfermarkt DuckDB)."""

    game_id: Column = Column("game_id", STR)
    competition_id: Column = Column("competition_id", STR)  # "GB1"
    season: Column = Column("season", STR)
    round: Column = Column("round", STR, nullable=True)
    date: Column = Column("date", DATE, nullable=True)
    home_club_id: Column = Column("home_club_id", INT32)
    away_club_id: Column = Column("away_club_id", INT32)
    home_club_goals: Column = Column("home_club_goals", INT32, nullable=True)
    away_club_goals: Column = Column("away_club_goals", INT32, nullable=True)
    home_club_position: Column = Column("home_club_position", INT32, nullable=True)
    away_club_position: Column = Column("away_club_position", INT32, nullable=True)
    home_club_manager_name: Column = Column("home_club_manager_name", STR, nullable=True)
    away_club_manager_name: Column = Column("away_club_manager_name", STR, nullable=True)
    stadium: Column = Column("stadium", STR, nullable=True)
    attendance: Column = Column("attendance", INT32, nullable=True)
    referee: Column = Column("referee", STR, nullable=True)
    url: Column = Column("url", STR, nullable=True)
    home_club_formation: Column = Column("home_club_formation", STR, nullable=True)
    away_club_formation: Column = Column("away_club_formation", STR, nullable=True)
    home_club_name: Column = Column("home_club_name", STR, nullable=True)
    away_club_name: Column = Column("away_club_name", STR, nullable=True)
    aggregate: Column = Column("aggregate", STR, nullable=True)
    competition_type: Column = Column("competition_type", STR, nullable=True)


@dataclass(frozen=True)
class TransfermarktCompetitionsRaw:
    """<transfermarkt.out_dir>/competitions.parquet (export of the Transfermarkt DuckDB)."""

    competition_id: Column = Column("competition_id", STR)
    competition_code: Column = Column("competition_code", STR, nullable=True)
    name: Column = Column("name", STR, nullable=True)
    sub_type: Column = Column("sub_type", STR, nullable=True)
    type: Column = Column("type", STR, nullable=True)  # "domestic_cup", "international_cup", ...
    country_id: Column = Column("country_id", INT32, nullable=True)
    country_name: Column = Column("country_name", STR, nullable=True)
    domestic_league_code: Column = Column("domestic_league_code", STR, nullable=True)
    confederation: Column = Column("confederation", STR, nullable=True)
    total_clubs: Column = Column("total_clubs", INT32, nullable=True)
    url: Column = Column("url", STR, nullable=True)


XG = UnderstatScheduleRaw()
PS = UnderstatPlayerRaw()
TM_GAMES = TransfermarktGamesRaw()
TM_COMPETITIONS = TransfermarktCompetitionsRaw()

type RawSchema = (
    UnderstatScheduleRaw
    | UnderstatPlayerRaw
    | TransfermarktGamesRaw
    | TransfermarktCompetitionsRaw
)

# Columns the clean pipeline reads from each raw source. Collection still checks
# every column of the schema; add a column here when the pipeline starts using it.
SELECTED: dict[type, tuple[Column, ...]] = {
    UnderstatScheduleRaw: (
        XG.league,
        XG.season,
        XG.game_id,
        XG.date,
        XG.home_team,
        XG.away_team,
        XG.home_goals,
        XG.away_goals,
        XG.home_xg,
        XG.away_xg,
        XG.is_result,
    ),
    UnderstatPlayerRaw: (
        PS.league,
        PS.season,
        PS.game,
        PS.team,
        PS.player,
        PS.game_id,
        PS.team_id,
        PS.player_id,
        PS.position,
        PS.minutes,
        PS.xg,
        PS.xa,
        PS.goals,
        PS.own_goals,
        PS.shots,
        PS.xg_chain,
        PS.xg_buildup,
        PS.assists,
        PS.key_passes,
        PS.yellow_cards,
        PS.red_cards,
    ),
    TransfermarktGamesRaw: (
        TM_GAMES.game_id,
        TM_GAMES.competition_id,
        TM_GAMES.date,
        TM_GAMES.home_club_id,
        TM_GAMES.away_club_id,
        TM_GAMES.home_club_goals,
        TM_GAMES.away_club_goals,
        TM_GAMES.home_club_formation,
        TM_GAMES.season,
        TM_GAMES.round,
        TM_GAMES.date,
        TM_GAMES.away_club_formation,
    ),
    TransfermarktCompetitionsRaw: (TM_COMPETITIONS.competition_id, TM_COMPETITIONS.type),
}


def selected(schema: RawSchema) -> list[Column]:
    """The columns of `schema` that the clean pipeline reads."""
    return list(SELECTED[type(schema)])


def raw_files(cfg: Settings) -> list[tuple[Path, RawSchema]]:
    """Every raw file the pipeline reads, paired with its schema."""
    files: list[tuple[Path, RawSchema]] = [
        (cfg.understat.schedule_file, XG),
        (cfg.transfermarkt.out_dir / "games.parquet", TM_GAMES),
        (cfg.transfermarkt.out_dir / "competitions.parquet", TM_COMPETITIONS),
    ]
    files += [(p, PS) for p in sorted(cfg.understat.out_dir.glob("*/*.parquet"))]
    return files


def _read(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=True)


def check_collected(path: Path, schema: RawSchema) -> None:
    """Collection check: raise if the file at `path` does not match every column of `schema`."""
    validate(_read(path), schema)


def read_selected(paths: Path | list[Path], schema: RawSchema) -> pl.DataFrame:
    """Read only the SELECTED columns of one CSV or one or more parquet files, then check them."""
    cols = selected(schema)
    names = [str(c) for c in cols]
    first = paths[0] if isinstance(paths, list) else paths
    if first.suffix == ".parquet":
        df = pl.read_parquet(paths, columns=names)
    else:
        df = pl.read_csv(first, columns=names, schema_overrides=dtypes(*cols))
    check(df, cols, type(schema).__name__)
    return df


def main() -> None:
    """Entry point `check_raw`: validate every raw file; exit 1 if any fails."""
    failed, ok = 0, 0
    for path, schema in raw_files(settings):
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            failed += 1
            continue
        try:
            check_collected(path, schema)
            ok += 1
        except ValueError as e:
            print(f"FAIL    {path}: {e}", flush=True)
            failed += 1
    print(f"{ok} ok, {failed} failed", flush=True)
    raise SystemExit(1 if failed else 0)


__all__ = [
    "PS",
    "SELECTED",
    "TM_COMPETITIONS",
    "TM_GAMES",
    "XG",
    "Column",
    "check_collected",
    "columns",
    "dtypes",
    "main",
    "raw_files",
    "read_selected",
    "selected",
    "validate",
]
