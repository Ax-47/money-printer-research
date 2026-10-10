from dataclasses import dataclass
from pathlib import Path

import polars as pl

from money_printer_research.clean_data.read import read_player_stat
from money_printer_research.config import Settings, UnderstatSettings
from money_printer_research.utils import SavesFrames


@dataclass(frozen=True)
class Frames(SavesFrames):
    clubs: pl.DataFrame
    matches: pl.DataFrame


def test_saves_frames_writes_one_csv_per_field(tmp_path: Path) -> None:
    frames = Frames(
        clubs=pl.DataFrame({"club_id": [1, 2]}), matches=pl.DataFrame({"game_id": [10]})
    )
    out = tmp_path / "new" / "dir"

    paths = frames.save(out)

    assert paths == [out / "clubs.csv", out / "matches.csv"]
    assert pl.read_csv(out / "clubs.csv").equals(frames.clubs)
    assert pl.read_csv(out / "matches.csv").equals(frames.matches)


def test_read_player_stat_without_files_is_an_err(tmp_path: Path) -> None:
    settings = Settings(understat=UnderstatSettings(out_dir=tmp_path))
    result = read_player_stat(settings)
    assert result.is_err()
    assert "No parquet files" in repr(result)
