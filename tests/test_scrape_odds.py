"""football-data.co.uk odds collection, with the download faked: no network."""

import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from money_printer_research.config import ModelingSettings, Settings, UnderstatSettings
from money_printer_research.data_collection import scrape_odds

LEAGUE = "ENG-Premier League"


def test_odds_url() -> None:
    assert (
        scrape_odds.odds_url("E0", 2024) == "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    )
    assert (
        scrape_odds.odds_url("SP1", 2009) == "https://www.football-data.co.uk/mmz4281/0910/SP1.csv"
    )


def test_odds_file_is_per_league_and_season(tmp_path: Path) -> None:
    assert scrape_odds.odds_file(tmp_path, LEAGUE, 2024) == tmp_path / LEAGUE / "2024-25.csv"


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake urlretrieve: records each URL, writes it to the file, 404s for 2026-27."""
    urls: list[str] = []

    def fake_urlretrieve(url: str, path: Path) -> None:
        if "/2627/" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        urls.append(url)
        path.write_text(url)

    monkeypatch.setattr(scrape_odds.urllib.request, "urlretrieve", fake_urlretrieve)
    return urls


def _settings(odds_dir: Path, end_year: int) -> Settings:
    return Settings(
        understat=UnderstatSettings(leagues=[LEAGUE], start_year=2023, end_year=end_year),
        modeling=ModelingSettings(odds_dir=odds_dir),
    )


def test_collect_odds_keeps_old_seasons_and_refreshes_the_last(
    fetched: list[str], tmp_path: Path
) -> None:
    settings = _settings(tmp_path, end_year=2026)
    scrape_odds.collect_odds(settings)
    assert sorted(p.name for p in (tmp_path / LEAGUE).iterdir()) == [
        "2023-24.csv",
        "2024-25.csv",
        "2025-26.csv",
    ]

    fetched.clear()
    scrape_odds.collect_odds(settings)
    assert fetched == [scrape_odds.odds_url("E0", 2025)]


def test_collect_odds_skips_a_season_not_published_yet(
    fetched: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scrape_odds.collect_odds(_settings(tmp_path, end_year=2027))
    assert not (tmp_path / LEAGUE / "2026-27.csv").exists()
    assert not (tmp_path / LEAGUE / "2026-27.csv.part").exists()
    assert "not published yet" in capsys.readouterr().out
