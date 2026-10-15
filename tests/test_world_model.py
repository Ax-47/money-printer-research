"""The modelling steps on tests/raw_world.py after clean_data.

Feature selection, the backtest, the learning curve, and the live poller predicting a
fixture with the saved model.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis import backtest, feature_importance, learning_curve
from money_printer_research.analysis.modeling import fit_final, save_model
from money_printer_research.clean_data import clean_data
from money_printer_research.config import Settings
from money_printer_research.live import espn, poller
from money_printer_research.live.poller import Fetchers, Predictor, refresh
from money_printer_research.live.store import Store
from tests.raw_world import SEASONS, STARTERS, write_raw


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = write_raw(tmp_path_factory.mktemp("world"))
    clean_data(settings)
    return settings


@pytest.fixture(scope="module")
def frame(world: Settings) -> pl.DataFrame:
    return feature_importance.load(world.clean.out_dir / "model_frame.csv")


def test_load_casts_every_feature_to_float(frame: pl.DataFrame) -> None:
    feats = feature_importance.columns(list(feature_importance.FEATURE_GROUPS))
    assert all(frame[c].dtype == pl.Float64 for c in feats)


def test_backtest_walk_forward_predicts_each_season_from_earlier_ones(frame: pl.DataFrame) -> None:
    pred = backtest.walk_forward(frame, [SEASONS[-1]])
    assert pred.height == frame.filter(pl.col("season_start") == SEASONS[-1]).height
    total = pred.select(pl.sum_horizontal("p_H", "p_D", "p_A")).to_series().to_numpy()
    assert np.allclose(total, 1.0)


def test_forward_selection_runs_until_no_group_is_left(frame: pl.DataFrame) -> None:
    # A gain floor of -1 accepts every group, so the loop ends by running out of them.
    # model_frame starts a season in (the first warms up Elo), so score the last one.
    steps = feature_importance.forward_selection(frame, min_gain=-1.0, seasons=(SEASONS[-1],))
    assert steps.height == len(feature_importance.FEATURE_GROUPS)
    assert steps["gain"][0] is None


def test_backtest_end_to_end(world: Settings) -> None:
    table = backtest.backtest(world)
    assert table.height > 0
    assert set(table["season_start"]) == {SEASONS[-1]}


def test_best_z_score_skips_cells_with_too_few_bets() -> None:
    profit = np.ones(5)
    assert backtest._best_z_score(profit, profit, [profit > 0], [0.0]) == -np.inf  # pyright: ignore[reportPrivateUsage]


def test_learning_curve_with_the_market_line(
    world: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(learning_curve, "TEST_FROM", SEASONS[-1])
    curve, refs, games = learning_curve.run(world)
    assert "market" in refs and games > 0
    with_market = learning_curve.render(curve, refs, games, "2026-10-15")
    without = learning_curve.render(curve, {"base_rates": refs["base_rates"]}, games, "2026-10-15")
    finding = "the best model is"
    assert finding in with_market and finding not in without
    path = world.modeling.learning_curve_file
    learning_curve.write(with_market, path)
    assert path.read_text(encoding="utf-8") == with_market


class _FirstSeason(datetime):
    """datetime whose now() is the start of the first season, so its fixtures are upcoming."""

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001, ANN206
        return datetime(SEASONS[0], 8, 1, tzinfo=tz)


def test_poller_predicts_a_fixture_with_the_saved_model(
    world: Settings, frame: pl.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    save_model(fit_final(frame), world.modeling.out_file)
    monkeypatch.setattr(poller, "datetime", _FirstSeason)
    predictor = Predictor(world)
    fixtures = predictor.fixtures()
    assert fixtures is not None
    first = fixtures.sort("date").row(0, named=True)
    kickoff = first["date"].replace(tzinfo=UTC)
    home, away = first["home_team"].title(), first["away_team"].title()

    def games_on(league: str, day: object) -> list[espn.Game]:
        if league != first["league"]:
            return []
        return [espn.Game("e1", league, kickoff, "pre", "3:00 PM", f"{home} FC", away, None, None)]

    def lineups(league: str, espn_id: str) -> list[espn.Starter]:
        return [
            espn.Starter(side, f"{team} Player {k + 1}", pos)
            for side, team in (("home", home), ("away", away))
            for k, pos in enumerate(STARTERS)
        ]

    store = Store(tmp_path / "live.sqlite")
    refresh(
        world, store, predictor, Fetchers(games_on, lineups), now=kickoff - timedelta(minutes=30)
    )
    game = store.game("e1")
    assert game is not None and game["game_id"] == first["game_id"]
    for source in ("fixture", "lineup"):
        assert sum(game[f"{source}_p_{k}"] for k in "HDA") == pytest.approx(1.0)
    assert all(p["player_id"] is not None for p in game["lineup"])
