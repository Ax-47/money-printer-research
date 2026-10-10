"""Season labels and the per-league, per-season output files the scrapers write."""

from collections.abc import Iterator
from pathlib import Path


def season_label(start_year: int) -> str:
    """Season name as soccerdata spells it: 2014 -> "2014-15"."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def understat_season_code(start_year: int) -> int:
    """Season as the Understat schedule stores it: 2014 -> 1415."""
    return (start_year % 100) * 100 + (start_year + 1) % 100


def season_files(
    out_dir: Path, leagues: list[str], start_year: int, end_year: int
) -> Iterator[tuple[str, str, Path]]:
    """Yield (league, season, file) for <out_dir>/<league>/<season>.parquet.

    end_year is exclusive. Each league folder is created before its first file is yielded.
    """
    for league in leagues:
        league_dir = out_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        for year in range(start_year, end_year):
            season = season_label(year)
            yield league, season, league_dir / f"{season}.parquet"
