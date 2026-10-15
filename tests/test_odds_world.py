"""market, strategies, staking and report on tests/odds_world.py's football-data files."""

import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from money_printer_research.analysis import report, staking, strategies
from money_printer_research.analysis.market import margin_methods, read_games
from money_printer_research.analysis.odds import read_prices
from money_printer_research.config import ModelingSettings, Settings
from tests.odds_world import SEASONS, TEAMS, write_odds

GAMES_PER_SEASON = len(TEAMS) * (len(TEAMS) - 1)


@pytest.fixture(scope="module")
def odds_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_odds(tmp_path_factory.mktemp("odds"))


@pytest.fixture(scope="module")
def suite(odds_dir: Path) -> dict[str, Any]:
    return strategies.run(Settings(modeling=ModelingSettings(odds_dir=odds_dir)))


def test_read_games_has_both_prices_and_the_result(odds_dir: Path) -> None:
    games = read_games(odds_dir)
    assert games.height == GAMES_PER_SEASON * len(SEASONS)
    assert set(games["result"]) <= {"H", "D", "A"}
    assert games.select(pl.col("^(early|close)_[HDA]$")).null_count().sum_horizontal()[0] == 0


def test_margin_methods_score_every_method_and_timing(odds_dir: Path) -> None:
    table = margin_methods(read_games(odds_dir))
    assert table.height == 6
    assert set(table["method"]) == {"proportional", "power", "shin"}
    assert table["log_loss"].is_between(0.5, 1.5).all()


def test_the_suite_bets_every_market_from_test_from(suite: dict[str, Any]) -> None:
    bets = suite["bets"]
    assert bets["season_start"].min() >= suite["test_from"]
    assert set(bets["strategy"]) >= {"model EV, Pinnacle early", "favourite"}
    assert suite["scores"]["q_value"].is_between(0, 1).all()
    assert suite["commit"] not in ("", None)


def test_implausible_prices_counts_per_price(odds_dir: Path) -> None:
    table = strategies.implausible_prices(read_prices(odds_dir))
    assert table.height > 0
    assert (table["dropped"] <= table["games"]).all()


def test_staking_skips_a_variant_with_no_bets(suite: dict[str, Any]) -> None:
    variant = suite["bets"]["variant"][0]
    out = staking.staking(suite["bets"], [variant, "no such variant"])
    assert set(out["replay"]["variant"]) == {variant}
    assert set(out["ruin"]["variant"]) == {variant}


def test_report_collects_every_section(odds_dir: Path) -> None:
    results = report.collect(Settings(modeling=ModelingSettings(odds_dir=odds_dir)))
    assert results["games"].height == GAMES_PER_SEASON * len(SEASONS)
    for key in ("margins", "forecasters", "scores", "staking", "time", "limits", "exchange"):
        assert key in results
    html = report.render(results)
    assert "<html" in html


def test_git_helpers_outside_a_checkout(tmp_path: Path) -> None:
    assert report._git("rev-parse", "HEAD", cwd=tmp_path) == "unknown"  # pyright: ignore[reportPrivateUsage]
    suite_file = tmp_path / "suite.toml"
    suite_file.write_text("")
    assert strategies.file_commit(suite_file) == "unknown"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    (tmp_path / "other").write_text("x")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "other"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "other"], cwd=tmp_path, check=True)
    assert strategies.file_commit(suite_file) == "uncommitted"


def test_git_helpers_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert report._git("status", cwd=tmp_path) == "unknown"  # pyright: ignore[reportPrivateUsage]
    assert strategies.file_commit(tmp_path / "x.toml") == "unknown"


def test_report_write_creates_the_folder(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b.html"
    report.write("<p>hi</p>", path)
    assert path.read_text(encoding="utf-8") == "<p>hi</p>"
