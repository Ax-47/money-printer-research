"""Checks for collected source files, and how raw models map to selected ones.

Every entity has a raw.py and a selected.py:

    raw.py       every column of a source file as collected. Collection checks every
                 column (check_collected), so a source that changes shape fails at
                 download time, not halfway through cleaning.
    selected.py  only the raw columns the clean pipeline reads, each copied line for
                 line from raw.py (same annotation, nullability and description).

Raw names are kept as they come from each source. Example values come from the
files as collected for 2014/15 to 2025/26.
"""

from pathlib import Path

import pandera.polars as pa
import polars as pl

from money_printer_research.schema.competition.raw import TransfermarktCompetitionsRaw
from money_printer_research.schema.competition.selected import TransfermarktCompetitions
from money_printer_research.schema.match.raw import TransfermarktGamesRaw, UnderstatScheduleRaw
from money_printer_research.schema.match.selected import TransfermarktGames, UnderstatSchedule
from money_printer_research.schema.player.raw import UnderstatPlayerRaw
from money_printer_research.schema.player.selected import UnderstatPlayer
from money_printer_research.schema.validation import validate_lazy

type RawSchema = (
    type[UnderstatScheduleRaw]
    | type[UnderstatPlayerRaw]
    | type[TransfermarktGamesRaw]
    | type[TransfermarktCompetitionsRaw]
)


def check_collected(path: Path, schema: RawSchema) -> None:
    """Raise ValueError if the file at `path` does not match every column of `schema`."""
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, try_parse_dates=True)
    validate_lazy(df, schema.to_schema(), schema.__name__)


# Raw model -> the model of the columns the clean pipeline reads.
# Collection checks the raw model (every column); when the pipeline starts using a
# column, copy its block from the raw model into the selected one.
SELECTED: dict[type[pa.DataFrameModel], type[pa.DataFrameModel]] = {
    UnderstatScheduleRaw: UnderstatSchedule,
    UnderstatPlayerRaw: UnderstatPlayer,
    TransfermarktGamesRaw: TransfermarktGames,
    TransfermarktCompetitionsRaw: TransfermarktCompetitions,
}


__all__ = ["SELECTED", "RawSchema", "check_collected"]
