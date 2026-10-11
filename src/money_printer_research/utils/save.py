from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def write_csv(frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path`, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


class SavesFrames:
    """Mixin for a frozen dataclass of DataFrames: save() writes each field as <name>.csv."""

    def save(self: "DataclassInstance", out_dir: Path) -> list[Path]:
        """Write each field to <out_dir>/<field name>.csv and return the paths."""
        paths: list[Path] = []
        for field in fields(self):
            path = out_dir / f"{field.name}.csv"
            write_csv(getattr(self, field.name), path)
            paths.append(path)
        return paths
