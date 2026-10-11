"""football-data.co.uk 1X2 odds, one CSV per league and season, for `backtest`.

Files go to <odds_dir>/<league>/<season>.csv, laid out like the player stats, so a reader
gets the Understat league from the folder and needs no division codes.
"""

import urllib.error
import urllib.request
from pathlib import Path

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.data_collection.seasons import season_label

DIVISIONS = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "GER-Bundesliga": "D1",
    "ITA-Serie A": "I1",
    "FRA-Ligue 1": "F1",
}
"""Understat league name to football-data.co.uk division code."""


def odds_url(division: str, start_year: int) -> str:
    """football-data.co.uk's CSV for one division and season: .../mmz4281/2425/E0.csv."""
    code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
    return f"https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"


def odds_file(odds_dir: Path, league: str, start_year: int) -> Path:
    """Where one league and season's odds go: <odds_dir>/<league>/2024-25.csv."""
    return odds_dir / league / f"{season_label(start_year)}.csv"


def _download(url: str, dest: Path) -> bool:
    """Fetch to <dest>.part and rename; False when football-data has no such file (404)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    tmp.replace(dest)
    return True


def collect_odds(settings: Settings = default_settings) -> None:
    """Each configured league's odds per season into settings.modeling.odds_dir.

    Files already there are kept, except the last season's, which grows as it is played.
    A season football-data has not published yet is skipped.
    """
    cfg = settings.understat
    for league in cfg.leagues:
        for year in range(cfg.start_year, cfg.end_year):
            path = odds_file(settings.modeling.odds_dir, league, year)
            if path.exists() and year != cfg.end_year - 1:
                continue
            if _download(odds_url(DIVISIONS[league], year), path):
                print("saved", path, flush=True)
            else:
                print("not published yet", path, flush=True)
