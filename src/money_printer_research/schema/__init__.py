"""Column names and dtypes of the clean outputs in settings.clean.out_dir, as pandera models.

The models live in modules by stage, and are re-exported here:

    schema/clean.py      single-source frames: xg, match_leagues (std_tm), player_stat
    schema/join.py       joined frames: player_matches, matches, clubs, cleaned_matches,
                         match_features, and PLAYER_SLOTS (the player slot columns)
    schema/formation.py  FormationCols, the encoded formation columns match_leagues and
                         cleaned_matches inherit, and GameFormationCols (games + formations)
    schema/elo.py        EloInputCols, EloCols, PredictionCols (Elo and the Poisson model)
    schema/fatigue.py    player and team fatigue
    schema/calendar.py   full-calendar schedule features

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
from money_printer_research.schema.calendar import COMP_TYPES, ClubCalendarCols, TeamScheduleCols
from money_printer_research.schema.clean import (
    CLEANED_MATCH_STAT,
    CLEANED_PLAYER_STAT,
    MATCH_LEAGUE,
    MatchLeaguesCols,
    PlayerStatCols,
    ScheduleCols,
)
from money_printer_research.schema.elo import OUTCOMES, EloCols, EloInputCols, PredictionCols
from money_printer_research.schema.fatigue import (
    FatigueInputCols,
    MatchSideCols,
    PlayerFatigueCols,
    TeamFatigueCols,
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
    MATCH_FEATURES,
    N_SLOTS,
    PLAYER_SLOT_FEATURES,
    PLAYER_SLOTS,
    PM,
    CleanedMatchCols,
    ClubMapCols,
    M,
    MatchCols,
    MatchFeaturesCols,
    PlayerMatchCols,
    match_features_schema,
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
    "CLEANED_MATCH",
    "CLEANED_MATCH_STAT",
    "CLEANED_PLAYER_STAT",
    "CLEAN_FILES",
    "CLUB_MAP",
    "COMP_TYPES",
    "FORMATION_COLUMNS",
    "GAME_FORMATION",
    "MATCH_FEATURES",
    "MATCH_LEAGUE",
    "N_SLOTS",
    "OUTCOMES",
    "OUTFIELD",
    "PLAYER_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PLAYER_STAT",
    "PM",
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
    "M",
    "MatchCols",
    "MatchFeaturesCols",
    "MatchLeaguesCols",
    "MatchSideCols",
    "PlayerFatigueCols",
    "PlayerMatchCols",
    "PlayerStatCols",
    "PredictionCols",
    "ScheduleCols",
    "TeamFatigueCols",
    "TeamScheduleCols",
    "TransfermarktCompetitions",
    "TransfermarktGames",
    "UnderstatPlayer",
    "UnderstatSchedule",
    "as_",
    "dtype",
    "main",
    "match_features_schema",
    "read_clean",
    "slot_column",
    "slot_columns",
    "validate_lazy",
]
