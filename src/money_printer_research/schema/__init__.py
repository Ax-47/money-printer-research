"""Column names and dtypes of the clean outputs in settings.clean.out_dir, as pandera models.

The models live in modules by stage, and are re-exported here:

    schema/clean.py      single-source frames: xg, std_tm, player_stat
    schema/join.py       joined frames: player_matches, matches, clubs, stat_matches
    schema/formation.py  FormationCols, the encoded formation columns std_tm and
                         stat_matches inherit, and GameFormationCols (games + formations)

(Raw files are in schema/raw.py, the raw columns the pipeline reads in
schema/selected.py.)

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
from money_printer_research.schema.clean import (
    CLEANED_MATCH_STAT,
    CLEANED_PLAYER_STAT,
    MATCH_LEAGUE,
    MatchLeaguesCols,
    PlayerStatCols,
    ScheduleCols,
)
from money_printer_research.schema.formation import (
    FORMATION_COLUMNS,
    GAME_FORMATION,
    OUTFIELD,
    SHAPE_PATTERN,
    FormationCols,
    GameFormationCols,
)
from money_printer_research.schema.join import (
    CLEANED_MATCH,
    CLUB_MAP,
    N_SLOTS,
    PLAYER_SLOT_FEATURES,
    PM,
    CleanedMatchCols,
    ClubMapCols,
    M,
    MatchCols,
    PlayerMatchCols,
    slot_column,
    slot_columns,
)
from money_printer_research.schema.raw import PLAYER_STAT
from money_printer_research.schema.selected import (
    SELECTED,
    TransfermarktCompetitions,
    TransfermarktGames,
    UnderstatPlayer,
    UnderstatSchedule,
)
from money_printer_research.schema.utils import as_, validate_lazy


def dtype(model: type[pa.DataFrameModel], name: str) -> pl.DataType:
    """Return the polars dtype of column `name` in `model`: .cast(dtype(ML, ML.date))."""
    return model.to_schema().columns[name].dtype.type


# File name (without .csv) -> the schema of that file.
CLEAN_FILES: dict[str, pa.DataFrameSchema] = {
    "xg": CLEANED_MATCH_STAT.to_schema(),
    "std_tm": MATCH_LEAGUE.to_schema(),
    "player_stat": CLEANED_PLAYER_STAT.to_schema(),
    "player_matches": PM.to_schema(),
    "matches": M.to_schema().add_columns(slot_columns()),
    "clubs": CLUB_MAP.to_schema(),
    "stat_matches": CLEANED_MATCH.to_schema(),
}


def read_clean(out_dir: Path, name: str) -> pl.DataFrame:
    """Read <out_dir>/<name>.csv with the schema's dtypes, then validate it."""
    schema = CLEAN_FILES[name]
    overrides = {n: col.dtype.type for n, col in schema.columns.items()}
    df = pl.read_csv(out_dir / f"{name}.csv", schema_overrides=overrides)
    return validate_lazy(df, schema, name)


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
    "CLEANED_MATCH_STAT",
    "CLEAN_FILES",
    "CLUB_MAP",
    "FORMATION_COLUMNS",
    "GAME_FORMATION",
    "MATCH_LEAGUE",
    "N_SLOTS",
    "OUTFIELD",
    "PLAYER_SLOT_FEATURES",
    "CLEANED_PLAYER_STAT",
    "PM",
    "SELECTED",
    "SHAPE_PATTERN",
    "CLEANED_MATCH",
    "ClubMapCols",
    "FormationCols",
    "GameFormationCols",
    "M",
    "MatchCols",
    "MatchLeaguesCols",
    "PlayerMatchCols",
    "PlayerStatCols",
    "ScheduleCols",
    "CleanedMatchCols",
    "TransfermarktCompetitions",
    "TransfermarktGames",
    "UnderstatPlayer",
    "UnderstatSchedule",
    "as_",
    "PLAYER_STAT",
    "dtype",
    "main",
    "read_clean",
    "slot_column",
    "slot_columns",
    "validate_lazy",
]
