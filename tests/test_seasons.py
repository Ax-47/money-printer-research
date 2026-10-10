from pathlib import Path

from money_printer_research.data_collection.seasons import (
    season_files,
    season_label,
    understat_season_code,
)


def test_season_label_spells_season_like_soccerdata() -> None:
    assert season_label(2014) == "2014-15"
    assert season_label(1999) == "1999-00"


def test_understat_season_code_joins_two_digit_years() -> None:
    assert understat_season_code(2014) == 1415
    assert understat_season_code(2009) == 910
    assert understat_season_code(1999) == 9900


def test_season_files_yields_one_file_per_league_season(tmp_path: Path) -> None:
    files = list(season_files(tmp_path, ["A", "B"], 2020, 2022))

    assert files == [
        ("A", "2020-21", tmp_path / "A" / "2020-21.parquet"),
        ("A", "2021-22", tmp_path / "A" / "2021-22.parquet"),
        ("B", "2020-21", tmp_path / "B" / "2020-21.parquet"),
        ("B", "2021-22", tmp_path / "B" / "2021-22.parquet"),
    ]
    assert (tmp_path / "A").is_dir()
    assert (tmp_path / "B").is_dir()


def test_season_files_end_year_is_exclusive(tmp_path: Path) -> None:
    assert list(season_files(tmp_path, ["A"], 2020, 2020)) == []
