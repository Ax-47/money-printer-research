import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.feature_importance import CLASSES, with_derived
from money_printer_research.analysis.modeling import (
    MODELS,
    calibration,
    calibration_error,
    compare,
    held_out,
    rps,
    score_grid_proba,
    scores,
)
from tests.test_feature_importance import _frame


def _games() -> pl.DataFrame:
    """test_feature_importance's frame (Elo drives the result) with goals that agree with it."""
    df = with_derived(_frame())
    rng = np.random.default_rng(1)
    result = df["full_time_result"].to_numpy()
    home = rng.poisson(1.2, df.height) + (result == "H")
    away = np.where(result == "D", home, rng.poisson(1.0, df.height) + (result == "A"))
    home = np.where(result == "A", np.minimum(home, away - 1).clip(0), home)
    away = np.where(result == "H", np.minimum(away, home - 1).clip(0), away)
    return df.with_columns(home_goals=pl.Series(home), away_goals=pl.Series(away))


def test_goals_agree_with_the_result() -> None:
    df = _games()
    sign = np.sign((df["home_goals"] - df["away_goals"]).to_numpy())
    assert (np.array(["A", "D", "H"])[sign + 1] == df["full_time_result"].to_numpy()).all()


def test_scores_of_a_perfect_and_a_uniform_forecast() -> None:
    y = np.array(["H", "D", "A"])
    perfect = (y[:, None] == np.array(CLASSES)[None, :]).astype(float)
    assert rps(y, perfect) == 0
    assert scores(y, perfect)["accuracy"] == 1
    assert scores(y, perfect)["brier"] == 0
    uniform = np.full((3, 3), 1 / 3)
    assert scores(y, uniform)["log_loss"] == pytest.approx(np.log(3))


def test_rps_charges_less_for_a_near_miss() -> None:
    y = np.array(["H"])
    draw = np.array([[0.0, 1.0, 0.0]])  # columns A, D, H
    away = np.array([[1.0, 0.0, 0.0]])
    assert rps(y, draw) < rps(y, away)


def test_score_grid_proba_is_symmetric_and_sums_to_one() -> None:
    proba = score_grid_proba(np.array([1.3, 2.0]), np.array([1.3, 0.5]))
    assert proba.sum(1) == pytest.approx([1, 1])
    a, _, h = proba[0]
    assert a == pytest.approx(h)
    assert proba[1, 2] > proba[1, 0]


def test_every_model_beats_the_prior_on_elo() -> None:
    df = _games()
    seasons = (2017, 2018)
    table = compare(df, groups=["elo"], seasons=seasons)
    assert table["model"].to_list() == list(MODELS)
    test = df.filter(pl.col("season_start").is_in(seasons))
    y = test["full_time_result"].to_numpy()
    shares = [float((y == c).mean()) for c in CLASSES]
    prior = scores(y, np.array([shares] * len(y)))["log_loss"]
    assert (table["log_loss"] < prior).all()


def test_held_out_scores_every_model() -> None:
    df = _games().with_columns(
        pl.col("season_start").replace_strict({2018: 2024}, default=pl.col("season_start"))
    )
    table = held_out(df, groups=["elo"])
    assert table["model"].to_list() == list(MODELS)
    assert table["log_loss"].is_finite().all()


def test_calibration_of_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([4, 3, 5], 20_000)
    y = np.array([rng.choice(CLASSES, p=p) for p in proba])
    table = calibration(y, proba)
    assert table.height == 3 * 10
    assert table["games"].sum() == 3 * len(y)
    assert (calibration_error(table)["ece"] < 0.02).all()
