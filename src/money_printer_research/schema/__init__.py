"""Column names and dtypes of the clean outputs in settings.clean.out_dir, as pandera models.

The models live in two modules by stage, and are re-exported here:

    schema/clean.py   single-source frames: xg, std_tm, player_stat
    schema/join.py    joined frames: player_matches, matches, clubs

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
    MATCH_LEAGUE,
    PLAYER_STAT,
    MatchLeaguesCols,
    PlayerStatCols,
    ScheduleCols,
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
from money_printer_research.schema.utils import as_, validate_lazy


def dtype(model: type[pa.DataFrameModel], name: str) -> pl.DataType:
    """The polars dtype of column `name` in `model`, e.g. .cast(dtype(TM, TM.date))."""
    return model.to_schema().columns[name].dtype.type


# File name (without .csv) -> the schema of that file.
CLEAN_FILES: dict[str, pa.DataFrameSchema] = {
    "xg": CLEANED_MATCH_STAT.to_schema(),
    "std_tm": MATCH_LEAGUE.to_schema(),
    "player_stat": PLAYER_STAT.to_schema(),
    "player_matches": PM.to_schema(),
    "matches": M.to_schema().add_columns(slot_columns()),
    "clubs": CLUB_MAP.to_schema(),
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
    "CLUB_MAP",
    "CLEANED_MATCH_STAT",
    "CLEAN_FILES",
    "M",
    "MatchLeaguesCols",
    "N_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PLAYER_STAT",
    "PM",
    "MATCH_LEAGUE",
    "ClubMapCols",
    "CleanedMatchCols",
    "CLEANED_MATCH",
    "MatchCols",
    "PlayerMatchCols",
    "PlayerStatCols",
    "ScheduleCols",
    "MatchLeaguesCols",
    "as_",
    "dtype",
    "main",
    "read_clean",
    "slot_column",
    "slot_columns",
]
