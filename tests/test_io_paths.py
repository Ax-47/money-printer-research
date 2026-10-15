"""The code that talks to the outside, with the network replaced.

ESPN's API, football-data downloads for elo_sensitivity, and the API's poller.
"""

import asyncio
import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from money_printer_research.analysis import elo_sensitivity
from money_printer_research.api import app as api_app
from money_printer_research.config import LiveSettings, ModelingSettings, Settings
from money_printer_research.data_collection.scrape_odds import DIVISIONS
from money_printer_research.live import espn

# --- ESPN -------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_espn_calls_build_urls_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    scoreboard = {
        "events": [
            {
                "id": "9",
                "date": "2026-10-17T14:00Z",
                "status": {"type": {"state": "pre", "shortDetail": "2:00 PM"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"displayName": "Everton"}, "score": "0"},
                            {"homeAway": "away", "team": {"displayName": "Chelsea"}, "score": "0"},
                        ]
                    }
                ],
            }
        ]
    }
    summary = {
        "rosters": [
            {"homeAway": "home", "roster": [{"starter": True, "athlete": {"displayName": "A"}}]}
        ]
    }

    def urlopen(request, timeout):  # noqa: ANN001, ANN202
        seen.append(request.full_url)
        body = scoreboard if "scoreboard" in request.full_url else summary
        return _Response(json.dumps(body).encode())

    monkeypatch.setattr(espn.urllib.request, "urlopen", urlopen)
    games = espn.games_on("ENG-Premier League", __import__("datetime").date(2026, 10, 17))
    assert [g.espn_id for g in games] == ["9"]
    starters = espn.lineups("ENG-Premier League", "9")
    assert [(s.side, s.name, s.position) for s in starters] == [("home", "A", "")]
    assert seen == [
        f"{espn.BASE}/eng.1/scoreboard?dates=20261017",
        f"{espn.BASE}/eng.1/summary?event=9",
    ]


# --- Elo sensitivity --------------------------------------------------------------


def _fd_season(path: Path, year: int) -> Path:
    teams = ["A", "B", "C", "D"]
    rows = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"]
    day = 1
    for h in teams:
        for a in teams:
            if h != a:
                rows.append(
                    f"E0,{day:02d}/09/{year % 100:02d},{h},{a},{(day * 7) % 4},{day % 3},X"
                )
                day += 1
    path.write_text("\n".join(rows) + "\n")
    return path


def test_sensitivity_over_a_small_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elo_sensitivity, "WARMUP_FROM", 2012)
    monkeypatch.setattr(elo_sensitivity, "TUNE_SEASONS", ["2013/14", "2014/15"])
    monkeypatch.setattr(elo_sensitivity, "K_GRID", [10.0, 20.0])
    monkeypatch.setattr(elo_sensitivity, "HOME_ADV_GRID", [0.0, 40.0])
    monkeypatch.setattr(elo_sensitivity, "REGRESS_GRID", [0.0, 0.2])
    monkeypatch.setattr(
        elo_sensitivity,
        "_season_file",
        lambda settings, league, year: _fd_season(tmp_path / f"{league}-{year}.csv", year),
    )
    matches = elo_sensitivity.load_history(Settings())
    assert matches.height == len(DIVISIONS) * 3 * 12
    table = elo_sensitivity.sensitivity(matches)
    assert table.height == 2 * 2 + 1  # the grid, then one regress value besides the default
    assert table["log_loss"].is_between(0.5, 1.5).all()


def test_season_file_downloads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(modeling=ModelingSettings(odds_dir=tmp_path))
    league = "ENG-Premier League"
    fetched: list[str] = []

    def download(url: str, dest: Path) -> bool:
        fetched.append(url)
        if "0506" in url:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x")
        return True

    monkeypatch.setattr(elo_sensitivity, "_download", download)
    path = elo_sensitivity._season_file(settings, league, 2004)  # pyright: ignore[reportPrivateUsage]
    assert path.read_text() == "x"
    elo_sensitivity._season_file(settings, league, 2004)  # pyright: ignore[reportPrivateUsage]
    assert len(fetched) == 1  # the second call finds the file
    with pytest.raises(FileNotFoundError):
        elo_sensitivity._season_file(settings, league, 2005)  # pyright: ignore[reportPrivateUsage]


def test_load_history_rejects_dates_outside_their_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(elo_sensitivity, "WARMUP_FROM", 2014)
    monkeypatch.setattr(elo_sensitivity, "TUNE_SEASONS", ["2014/15"])
    # La Liga's file is dated 2030 while it is read as the 2014/15 season.
    monkeypatch.setattr(
        elo_sensitivity,
        "_season_file",
        lambda settings, league, year: _fd_season(
            tmp_path / f"{league}.csv", 2030 if "Liga" in league else year
        ),
    )
    with pytest.raises(ValueError, match="outside their season"):
        elo_sensitivity.load_history(Settings())


# --- The API's poller -------------------------------------------------------------


def test_lifespan_polls_until_the_app_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def refresh(settings, store, predictor):  # noqa: ANN001, ANN202
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("ESPN down")  # logged; the loop goes on
        return 0

    real_sleep = asyncio.sleep
    monkeypatch.setattr(api_app, "refresh", refresh)
    monkeypatch.setattr(api_app.asyncio, "sleep", lambda s: real_sleep(0.01))
    settings = Settings(live=LiveSettings(enabled=True, db_file=tmp_path / "live.sqlite"))
    with TestClient(api_app.create_app(settings)) as client:
        deadline = time.monotonic() + 5
        while len(calls) < 3 and time.monotonic() < deadline:
            client.get("/health")
            time.sleep(0.02)
    assert len(calls) >= 3
    assert (tmp_path / "live.sqlite").exists()
