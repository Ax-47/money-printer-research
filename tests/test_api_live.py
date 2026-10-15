from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from money_printer_research.api import create_app
from money_printer_research.config import CleanSettings, LiveSettings, Settings
from money_printer_research.live.store import Store

LEAGUE = "ENG-Premier League"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    # Season 2026: chelsea beat everton 2-0, drew fulham 1-1; everton beat fulham 3-1.
    # Season 2025 is older and must not count.
    pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "date": ["2025-09-01", "2026-08-20", "2026-08-27", "2026-09-03"],
            "league": [LEAGUE] * 4,
            "season_start": [2025, 2026, 2026, 2026],
            "home_team": ["everton", "chelsea", "fulham", "everton"],
            "away_team": ["chelsea", "everton", "chelsea", "fulham"],
            "home_goals": [5, 2, 1, 3],
            "away_goals": [0, 0, 1, 1],
            "home_xg": [3.0, 1.8, 1.2, 2.1],
            "away_xg": [0.2, 0.4, 0.9, 0.7],
        }
    ).write_csv(tmp_path / "cleaned_matches.csv")
    pl.DataFrame(
        {
            "season": ["2026/27", "2026/27"],
            "match_date": ["2026-08-20", "2026-08-27"],
            "home_team": ["chelsea", "fulham"],
            "away_team": ["everton", "chelsea"],
            "elo_home": [1550.0, 1480.0],
            "elo_away": [1500.0, 1560.0],
        }
    ).write_csv(tmp_path / "elo.csv")
    settings = Settings(
        clean=CleanSettings(out_dir=tmp_path),
        live=LiveSettings(enabled=False, db_file=tmp_path / "live.sqlite"),
    )
    store = Store(settings.live.db_file)
    today = datetime.now(UTC).replace(hour=14, minute=0, second=0, microsecond=0)
    for espn_id, kickoff, league in (
        ("1", today, LEAGUE),
        ("2", today, "ESP-La Liga"),
        ("3", today.replace(year=today.year - 1), LEAGUE),
    ):
        store.upsert_game(
            {
                "espn_id": espn_id,
                "league": league,
                "kickoff": kickoff.isoformat(),
                "state": "in",
                "detail": "67'",
                "espn_home": "Chelsea",
                "espn_away": "Everton",
                "home_team": "chelsea",
                "away_team": "everton",
                "game_id": 10,
                "home_score": 1,
                "away_score": 0,
            }
        )
    store.set_lineup("1", [{"side": "home", "name": "Cole Palmer", "position": "AM"}])
    store.set_prediction("1", "lineup", {"p_H": 0.6, "p_D": 0.25, "p_A": 0.15})
    store.close()
    return TestClient(create_app(settings))


def test_league_table_ranks_the_current_season(client: TestClient) -> None:
    table = client.get("/teams", params={"league": LEAGUE}).json()
    assert [(r["rank"], r["team"], r["points"]) for r in table] == [
        (1, "chelsea", 4),
        (2, "everton", 3),
        (3, "fulham", 1),
    ]
    assert table[0]["goals_for"] == 3 and table[0]["played"] == 2


def test_teams_lists_every_team(client: TestClient) -> None:
    assert [r["team"] for r in client.get("/teams").json()] == ["chelsea", "everton", "fulham"]


def test_team_stats(client: TestClient) -> None:
    body = client.get("/teams/chelsea").json()
    assert body["season_start"] == 2026
    assert body["form"] == "DW"  # latest first
    assert body["table"]["rank"] == 1
    assert {v["venue"]: v["points"] for v in body["by_venue"]} == {"home": 3, "away": 1}
    assert body["elo_history"] == [
        {"match_date": "2026-08-20", "elo": 1550.0},
        {"match_date": "2026-08-27", "elo": 1560.0},
    ]


def test_unknown_team_and_league_are_404(client: TestClient) -> None:
    assert client.get("/teams/nobody").status_code == 404
    assert client.get("/teams", params={"league": "nowhere"}).status_code == 404


def test_live_lists_todays_games(client: TestClient) -> None:
    games = client.get("/live").json()
    assert [g["espn_id"] for g in games] == ["1", "2"]  # last year's game left out
    assert games[0]["lineup_p_H"] == 0.6 and games[0]["fixture_p_H"] is None
    assert [g["has_lineup"] for g in games] == [1, 0]
    assert [g["espn_id"] for g in client.get("/live", params={"league": LEAGUE}).json()] == ["1"]


def test_live_game_has_its_lineup(client: TestClient) -> None:
    body = client.get("/live/1").json()
    assert (body["home_score"], body["detail"]) == (1, "67'")
    assert body["lineup"][0]["name"] == "Cole Palmer"
    assert client.get("/live/nope").status_code == 404


def test_app_without_live_access_makes_no_database(tmp_path: Path) -> None:
    db = tmp_path / "never.sqlite"
    settings = Settings(live=LiveSettings(enabled=False, db_file=db))
    TestClient(create_app(settings)).get("/health")
    assert not db.exists()


@pytest.mark.parametrize("page", ["live.html", "teams.html"])
def test_pages_are_served_and_linked_from_every_page(client: TestClient, page: str) -> None:
    assert client.get(f"/{page}").status_code == 200
    for other in ("/", "/backtest.html", "/analysis.html", "/live.html", "/teams.html"):
        assert f'href="{page}"' in client.get(other).text
