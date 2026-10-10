from pathlib import Path
from typing import cast

import pandas as pd
import pandera.polars as pa
import polars as pl
import pytest

from money_printer_research.data_collection.checked import save_checked
from money_printer_research.schema.collected import RawSchema, check_collected
from money_printer_research.schema.validation import validate_lazy


class Tiny(pa.DataFrameModel):
    """A two-column stand-in for the raw source schemas."""

    game_id: int = pa.Field(ge=0)
    team: str


TINY = cast(RawSchema, Tiny)


def _good() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [1, 2], "team": ["Arsenal", "Chelsea"]})


def _bad() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [-1, -2], "team": ["Arsenal", "Chelsea"]})


def test_validate_lazy_returns_frame_when_valid() -> None:
    df = pl.DataFrame({"game_id": [1], "team": ["Arsenal"]})
    assert validate_lazy(df, Tiny.to_schema(), "Tiny").equals(df)


def test_validate_lazy_reports_every_failure_in_one_value_error() -> None:
    df = pl.DataFrame({"game_id": [-1, -2], "team": ["Arsenal", "Chelsea"]})
    with pytest.raises(ValueError, match="^Tiny:") as info:
        validate_lazy(df, Tiny.to_schema(), "Tiny")
    # Both bad rows are listed, not just the first.
    assert "-1" in str(info.value)
    assert "-2" in str(info.value)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_check_collected_reads_parquet_and_csv(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"games{suffix}"
    if suffix == ".parquet":
        _good().to_parquet(path)
    else:
        _good().to_csv(path, index=False)
    check_collected(path, TINY)


def test_check_collected_raises_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "games.csv"
    _bad().to_csv(path, index=False)
    with pytest.raises(ValueError, match="Tiny"):
        check_collected(path, TINY)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_save_checked_writes_valid_frame(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"games{suffix}"
    save_checked(_good(), out, TINY)

    assert out.exists()
    assert not (tmp_path / f"games.part{suffix}").exists()
    saved = pl.read_parquet(out) if suffix == ".parquet" else pl.read_csv(out)
    assert saved["team"].to_list() == ["Arsenal", "Chelsea"]


def test_save_checked_leaves_no_file_when_check_fails(tmp_path: Path) -> None:
    out = tmp_path / "games.parquet"
    with pytest.raises(ValueError):
        save_checked(_bad(), out, TINY)
    assert list(tmp_path.iterdir()) == []


def test_save_checked_keeps_previous_file_when_check_fails(tmp_path: Path) -> None:
    out = tmp_path / "games.csv"
    _good().to_csv(out, index=False)
    with pytest.raises(ValueError):
        save_checked(_bad(), out, TINY)
    assert pl.read_csv(out)["game_id"].to_list() == [1, 2]
