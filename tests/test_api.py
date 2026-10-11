import datetime
import itertools
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from pandera.typing.polars import DataFrame

from money_printer_research.api import create_app
from money_printer_research.config import CleanSettings, ModelingSettings, Settings
from money_printer_research.features.elo import get_elo
from money_printer_research.schema import EloInputCols

TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham"]
SEASONS = ["2014/15", "2015/16", "2016/17"]


def _elo_input() -> DataFrame[EloInputCols]:
    """Every pair of TEAMS home and away in each of SEASONS, with seeded random scores."""
    rng = np.random.default_rng(0)
    rows = [
        (season, home, away)
        for season in SEASONS
        for home, away in itertools.permutations(TEAMS, 2)
    ]
    home_goals = rng.poisson(1.5, len(rows)).tolist()
    away_goals = rng.poisson(1.1, len(rows)).tolist()
    start = datetime.date(2014, 8, 1)
    return DataFrame[EloInputCols](
        pl.DataFrame(
            {
                "season": [r[0] for r in rows],
                "match_date": [start + datetime.timedelta(days=i) for i in range(len(rows))],
                "home_team": [r[1] for r in rows],
                "away_team": [r[2] for r in rows],
                "full_time_home_goals": home_goals,
                "full_time_away_goals": away_goals,
                "full_time_result": [
                    "H" if h > a else "A" if a > h else "D"
                    for h, a in zip(home_goals, away_goals, strict=True)
                ],
            }
        )
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    get_elo(_elo_input()).write_csv(tmp_path / "elo.csv")
    pl.DataFrame({"game_id": [1, 2], "xg": [0.5, float("nan")]}).write_csv(
        tmp_path / "fatigue.csv"
    )
    upcoming = tmp_path / "predictions" / "upcoming.csv"
    upcoming.parent.mkdir()
    pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "date": ["2026-10-17 14:00:00", "2026-10-17 16:30:00", "2026-10-18 19:00:00"],
            "league": ["ENG-Premier League", "ENG-Premier League", "ESP-La Liga"],
            "home_team": ["Arsenal", "Everton", "Barcelona"],
            "away_team": ["Chelsea", "Arsenal", "Sevilla"],
            "p_H": [0.5, 0.3, 0.6],
            "p_D": [0.3, 0.3, 0.25],
            "p_A": [0.2, 0.4, 0.15],
        }
    ).write_csv(upcoming)
    settings = Settings(
        clean=CleanSettings(out_dir=tmp_path), modeling=ModelingSettings(predictions_file=upcoming)
    )
    return TestClient(create_app(settings))


def test_health_counts_tables(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "tables": 2}


def test_lists_tables(client: TestClient) -> None:
    assert client.get("/tables").json() == ["elo", "fatigue"]


def test_table_is_paged(client: TestClient) -> None:
    body = client.get("/tables/elo", params={"limit": 5, "offset": 10}).json()
    assert body["total"] == len(SEASONS) * 12
    assert len(body["rows"]) == 5
    assert body["rows"][0]["match_date"] == "2014-08-11"


def test_nan_is_served_as_null(client: TestClient) -> None:
    rows = client.get("/tables/fatigue").json()["rows"]
    assert rows[1]["xg"] is None


def test_unknown_table_is_404(client: TestClient) -> None:
    assert client.get("/tables/missing").status_code == 404
    assert client.get("/tables/..%2Fsettings").status_code == 404


def test_first_season_is_not_predictable(client: TestClient) -> None:
    assert client.get("/predictions/seasons").json() == SEASONS[1:]
    assert client.get("/predictions", params={"season": SEASONS[0]}).status_code == 404


def test_predictions_are_probabilities(client: TestClient) -> None:
    rows = client.get("/predictions", params={"season": "2016/17"}).json()
    assert len(rows) == 12
    for row in rows:
        assert row["season"] == "2016/17"
        assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0)


def test_predictions_filter_by_team(client: TestClient) -> None:
    rows = client.get("/predictions", params={"season": "2016/17", "team": "Arsenal"}).json()
    assert len(rows) == 6
    assert all("Arsenal" in (r["home_team"], r["away_team"]) for r in rows)


def test_metrics(client: TestClient) -> None:
    body = client.get("/predictions/metrics", params={"season": "2016/17"}).json()
    assert body["n"] == 12
    assert 0 <= body["accuracy"] <= 1
    assert body["log_loss"] > 0


def test_upcoming_filters_by_league_and_team(client: TestClient) -> None:
    assert len(client.get("/upcoming").json()) == 3
    rows = client.get("/upcoming", params={"league": "ENG-Premier League"}).json()
    assert [r["game_id"] for r in rows] == [1, 2]
    rows = client.get("/upcoming", params={"team": "Arsenal"}).json()
    assert [r["game_id"] for r in rows] == [1, 2]
    rows = client.get("/upcoming", params={"league": "ESP-La Liga", "team": "Arsenal"}).json()
    assert rows == []


def test_upcoming_without_predict_run_is_404(tmp_path: Path) -> None:
    settings = Settings(modeling=ModelingSettings(predictions_file=tmp_path / "none.csv"))
    response = TestClient(create_app(settings)).get("/upcoming")
    assert response.status_code == 404
    assert "uv run predict" in response.json()["detail"]


def test_index_serves_the_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'fetch("upcoming")' in response.text
