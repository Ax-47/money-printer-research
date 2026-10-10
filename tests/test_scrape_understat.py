"""Understat scrapers, with soccerdata replaced by a fake: no test touches the network."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import polars as pl
import pytest

from money_printer_research.config import Settings, UnderstatSettings
from money_printer_research.data_collection import scrape_player_stat, scrape_schedule
from money_printer_research.schema.collected import RawSchema
from money_printer_research.schema.match.raw import UnderstatScheduleRaw
from money_printer_research.schema.player.raw import UnderstatPlayerRaw


class FakeUnderstat:
    """Stands in for soccerdata.Understat and records how it was built."""

    created: list[dict[str, Any]] = []
    # Match ids whose single-match read fails the way soccerdata does on empty rosters.
    no_rosters: set[int] = set()
    # When True, the whole-season read fails, forcing the match-by-match fallback.
    season_read_fails = False
    # When True, building the reader raises, like a network error.
    broken = False

    def __init__(self, leagues: Any, seasons: list[str], data_dir: Path) -> None:
        if FakeUnderstat.broken:
            raise ConnectionError("no network in tests")
        self.leagues = leagues
        self.seasons = seasons
        FakeUnderstat.created.append(
            {"leagues": leagues, "seasons": seasons, "data_dir": data_dir}
        )

    def read_schedule(self) -> pd.DataFrame:
        """Three matches; the last has no result yet."""
        return pd.DataFrame({"game_id": [10, 11, 12], "is_result": [True, True, False]}).set_index(
            "game_id"
        )

    def read_player_match_stats(self, match_id: int | None = None) -> pd.DataFrame:
        """The whole season, or one match when match_id is given."""
        if match_id is None:
            if FakeUnderstat.season_read_fails:
                raise AttributeError("'NoneType' object has no attribute 'items'")
            return pd.DataFrame({"game_id": [10, 11], "player": ["a", "b"]}).set_index("game_id")
        if match_id in FakeUnderstat.no_rosters:
            raise AttributeError("'NoneType' object has no attribute 'items'")
        return pd.DataFrame({"game_id": [match_id], "player": [f"p{match_id}"]}).set_index(
            "game_id"
        )


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[pd.DataFrame, Path, RawSchema]]:
    """Fake soccerdata in both scrapers and record every save_checked call."""
    FakeUnderstat.created = []
    FakeUnderstat.no_rosters = set()
    FakeUnderstat.season_read_fails = False
    FakeUnderstat.broken = False
    fake_sd = SimpleNamespace(Understat=FakeUnderstat)
    monkeypatch.setattr(scrape_player_stat, "sd", fake_sd)
    monkeypatch.setattr(scrape_schedule, "sd", fake_sd)

    calls: list[tuple[pd.DataFrame, Path, RawSchema]] = []

    def record(df: pd.DataFrame, out: Path, schema: RawSchema) -> None:
        calls.append((df, out, schema))
        out.touch()

    monkeypatch.setattr(scrape_player_stat, "save_checked", record)
    monkeypatch.setattr(scrape_schedule, "save_checked", record)
    return calls


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        understat=UnderstatSettings(
            leagues=["ENG-Premier League"],
            start_year=2020,
            end_year=2022,
            out_dir=tmp_path / "player_stats",
            schedule_file=tmp_path / "xg_schedule.csv",
            cache_dir=tmp_path / "cache",
        )
    )


def test_collect_player_stats_saves_one_file_per_season(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    scrape_player_stat.collect_player_stats(_settings(tmp_path))

    league_dir = tmp_path / "player_stats" / "ENG-Premier League"
    assert [out for _, out, _ in saved] == [
        league_dir / "2020-21.parquet",
        league_dir / "2021-22.parquet",
    ]
    assert all(schema is UnderstatPlayerRaw for _, _, schema in saved)
    # The index is reset, so game_id is saved as a column.
    assert "game_id" in saved[0][0].columns
    assert [c["seasons"] for c in FakeUnderstat.created] == [["2020-21"], ["2021-22"]]
    assert FakeUnderstat.created[0]["data_dir"] == tmp_path / "cache"


def test_collect_player_stats_skips_existing_files(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    league_dir = tmp_path / "player_stats" / "ENG-Premier League"
    league_dir.mkdir(parents=True)
    (league_dir / "2020-21.parquet").touch()

    scrape_player_stat.collect_player_stats(_settings(tmp_path))

    assert [out.name for _, out, _ in saved] == ["2021-22.parquet"]


def test_collect_player_stats_falls_back_to_match_by_match(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    FakeUnderstat.season_read_fails = True
    FakeUnderstat.no_rosters = {11}

    scrape_player_stat.collect_player_stats(_settings(tmp_path))

    # Only finished matches are read (12 has no result) and 11 is skipped.
    assert saved[0][0]["game_id"].tolist() == [10]


def test_collect_player_stats_keeps_going_after_a_failed_season(
    tmp_path: Path,
    saved: list[tuple[pd.DataFrame, Path, RawSchema]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeUnderstat.broken = True

    scrape_player_stat.collect_player_stats(_settings(tmp_path))

    assert saved == []
    assert capsys.readouterr().out.count("FAILED") == 2


def test_collect_schedule_reads_every_season_at_once(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    scrape_schedule.collect_schedule(_settings(tmp_path))

    assert FakeUnderstat.created == [
        {
            "leagues": ["ENG-Premier League"],
            "seasons": ["2020-21", "2021-22"],
            "data_dir": tmp_path / "cache",
        }
    ]
    [(df, out, schema)] = saved
    assert out == tmp_path / "xg_schedule.csv"
    assert schema is UnderstatScheduleRaw
    assert "game_id" in df.columns


def test_collect_schedule_skips_when_every_season_is_saved(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    pl.DataFrame({"season": [2021, 2122]}).write_csv(tmp_path / "xg_schedule.csv")

    scrape_schedule.collect_schedule(_settings(tmp_path))

    assert saved == []
    assert FakeUnderstat.created == []


def test_collect_schedule_refreshes_when_a_season_is_missing(
    tmp_path: Path, saved: list[tuple[pd.DataFrame, Path, RawSchema]]
) -> None:
    pl.DataFrame({"season": [2021]}).write_csv(tmp_path / "xg_schedule.csv")

    scrape_schedule.collect_schedule(_settings(tmp_path))

    assert len(saved) == 1
