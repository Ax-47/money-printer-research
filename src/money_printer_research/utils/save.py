from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class SavesFrames:
    """Mixin for a frozen dataclass of DataFrames: save() writes each field as <name>.csv."""

    def save(self: "DataclassInstance", out_dir: Path) -> list[Path]:
        """Write each field to <out_dir>/<field name>.csv and return the paths."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for field in fields(self):
            frame: pl.DataFrame = getattr(self, field.name)
            path = out_dir / f"{field.name}.csv"
            frame.write_csv(path)
            paths.append(path)
        return paths
