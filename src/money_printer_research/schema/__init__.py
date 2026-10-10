"""Column names, dtypes and descriptions of every frame, as pandera models, by entity.

    match/        one row per match: the Understat schedule, Transfermarkt games,
                  and the pre-match features joined onto them
    player/       one row per player per match: Understat player stats and load
    club/         the Transfermarkt club -> Understat team map, and each club's calendar
    competition/  Transfermarkt competitions and how they are grouped (COMP_TYPES)

Inside each entity, one module per stage, in pipeline order (an entity has only
the stages it needs):

    raw.py        every column of a collected file; collected.py explains raw vs selected
    selected.py   the raw columns the clean pipeline reads
    clean.py      cleaned frames, written as CSV to settings.clean.out_dir
    features.py   pre-match features built from the clean frames

Everything is re-exported here, so `from money_printer_research.schema import ...`
works for any model.

Every column is F[...] (pandera FieldType) with a description, so
pl.col(CLEANED_MATCH_STAT.season) works and pyright types it as str. The dtypes
are the ones the pipeline produces in memory. CSV loses some of them (season
"1516" comes back as an integer), so read the CSVs with read_clean(), which applies
the schema's dtypes, then validates.
"""

from pathlib import Path

import pandera.polars as pa
import polars as pl

from money_printer_research.config import settings
from money_printer_research.schema.club import CLUB_MAP, ClubCalendarCols, ClubMapCols
from money_printer_research.schema.collected import SELECTED, RawSchema, check_collected
from money_printer_research.schema.competition import (
    COMP_TYPES,
    LEAGUE_INFO,
    TransfermarktCompetitions,
    TransfermarktCompetitionsRaw,
)
from money_printer_research.schema.match import (
    CLEANED_MATCH,
    CLEANED_MATCH_STAT,
    FORMATION_COLUMNS,
    GAME_FORMATION,
    MATCH_FEATURES,
    MATCH_INFO,
    MATCH_LEAGUE,
    MATCH_STAT,
    N_SLOTS,
    OUTCOMES,
    OUTFIELD,
    PLAYER_SLOT_FEATURES,
    PLAYER_SLOTS,
    SHAPE_PATTERN,
    CleanedMatchCols,
    EloCols,
    EloInputCols,
    FormationCols,
    GameFormationCols,
    MatchFeaturesCols,
    MatchLeaguesCols,
    MatchSideCols,
    PredictionCols,
    ScheduleCols,
    TeamFatigueCols,
    TeamScheduleCols,
    TransfermarktGames,
    TransfermarktGamesRaw,
    UnderstatSchedule,
    UnderstatScheduleRaw,
    match_features_schema,
    slot_column,
    slot_columns,
)
from money_printer_research.schema.player import (
    CLEANED_PLAYER_STAT,
    PLAYER_STAT,
    FatigueInputCols,
    PlayerFatigueCols,
    PlayerStatCols,
    UnderstatPlayer,
    UnderstatPlayerRaw,
)
from money_printer_research.schema.validation import validate_lazy


def dtype(model: type[pa.DataFrameModel], name: str) -> pl.DataType:
    """Return the polars dtype of column `name` in `model`: .cast(dtype(ML, ML.date))."""
    return model.to_schema().columns[name].dtype.type


# File name (without .csv) -> the schema of that file: the fields of Cleaned and
# ExactFeatures in clean_data, which SavesFrames.save writes as <field name>.csv.
CLEAN_FILES: dict[str, pa.DataFrameSchema] = {
    # Cleaned
    "match_leagues": MATCH_LEAGUE.to_schema(),
    "clubs": CLUB_MAP.to_schema(),
    "cleaned_matches": CLEANED_MATCH.to_schema(),
    "player_stat": CLEANED_PLAYER_STAT.to_schema(),
    # ExactFeatures
    "elo": EloCols.to_schema(),
    "fatigue": TeamFatigueCols.to_schema(),
    "schedule": TeamScheduleCols.to_schema(),
    "players": PLAYER_SLOTS,
    "match_features": match_features_schema(),
}


def read_clean(out_dir: Path, name: str) -> pl.DataFrame:
    """Read <out_dir>/<name>.csv with the schema's dtypes, then validate it."""
    schema = CLEAN_FILES[name]
    overrides = {n: col.dtype.type for n, col in schema.columns.items()}
    df = pl.read_csv(out_dir / f"{name}.csv", schema_overrides=overrides)
    return validate_lazy(df, schema, name)


def check_clean() -> None:
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
    "CLEANED_MATCH",
    "CLEANED_MATCH_STAT",
    "CLEANED_PLAYER_STAT",
    "CLEAN_FILES",
    "CLUB_MAP",
    "COMP_TYPES",
    "FORMATION_COLUMNS",
    "GAME_FORMATION",
    "LEAGUE_INFO",
    "MATCH_FEATURES",
    "MATCH_INFO",
    "MATCH_LEAGUE",
    "MATCH_STAT",
    "N_SLOTS",
    "OUTCOMES",
    "OUTFIELD",
    "PLAYER_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PLAYER_STAT",
    "SELECTED",
    "SHAPE_PATTERN",
    "CleanedMatchCols",
    "ClubCalendarCols",
    "ClubMapCols",
    "EloCols",
    "EloInputCols",
    "FatigueInputCols",
    "FormationCols",
    "GameFormationCols",
    "MatchFeaturesCols",
    "MatchLeaguesCols",
    "MatchSideCols",
    "PlayerFatigueCols",
    "PlayerStatCols",
    "PredictionCols",
    "RawSchema",
    "ScheduleCols",
    "TeamFatigueCols",
    "TeamScheduleCols",
    "TransfermarktCompetitions",
    "TransfermarktCompetitionsRaw",
    "TransfermarktGames",
    "TransfermarktGamesRaw",
    "UnderstatPlayer",
    "UnderstatPlayerRaw",
    "UnderstatSchedule",
    "UnderstatScheduleRaw",
    "check_clean",
    "check_collected",
    "dtype",
    "match_features_schema",
    "read_clean",
    "slot_column",
    "slot_columns",
    "validate_lazy",
]
