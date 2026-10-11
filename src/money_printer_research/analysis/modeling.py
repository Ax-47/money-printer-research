"""H/D/A models on the selected features (feature_importance.SELECTED), compared and checked.

Every model is fitted and scored the way feature selection was: walk-forward over
VALID_SEASONS (each season predicted by a model fitted on the seasons before it), and
once on the held-out seasons from TEST_FROM, fitted on everything before.

The baseline is feature_importance.make_model, a multinomial logistic regression. It is
compared with the same model recalibrated (sigmoid, 5-fold on the training seasons),
gradient-boosted trees, and a goals model: one Poisson regression per side for the
number of goals, turned into H/D/A by summing the independent score grid.

Scores are log loss (what selection minimised), the ranked probability score (the usual
one for football: it treats H, D, A as ordered, so calling a home win a draw costs less
than calling it an away win), the multiclass Brier score and accuracy. calibration bins
each class's predicted probability by decile and compares it with how often it happened.

After the comparison, fit_final refits the logistic baseline on every season (the held-out
ones too: they have been scored, and the newest games matter most) and save_model writes
it with joblib to settings.modeling.out_file. load_model reads it back; SavedModel.predict
takes model_frame rows and returns P(A), P(D), P(H) per game.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from money_printer_research.analysis.feature_importance import (
    CLASSES,
    SELECTED,
    TARGET,
    TEST_FROM,
    VALID_SEASONS,
    columns,
    load,
    make_model,
    with_derived,
)
from money_printer_research.config import settings

type Model = Callable[[pl.DataFrame, pl.DataFrame, list[str]], np.ndarray]
"""(train, test, feature columns) -> test probabilities, one column per CLASSES."""

MAX_GOALS = 10
"""Scores above this per side are left out of the Poisson grid (then renormalised)."""


def _fit_predict(
    estimator: Any, train: pl.DataFrame, test: pl.DataFrame, cols: list[str]
) -> np.ndarray:
    estimator.fit(train.select(cols).to_numpy(), train[TARGET].to_numpy())
    return estimator.predict_proba(test.select(cols).to_numpy())


def logistic(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """The baseline: feature_importance.make_model."""
    return _fit_predict(make_model(), train, test, cols)


def logistic_sigmoid(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """The baseline with a sigmoid recalibration per class, fitted by 5-fold CV."""
    return _fit_predict(
        CalibratedClassifierCV(make_model(), method="sigmoid", cv=5), train, test, cols
    )


def boosted_trees(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Shallow, slow gradient boosting; trees take the nulls as they are.

    On more than 10,000 games scikit-learn stops early on a random 10% of the training
    games; random_state fixes that split, so the scores are the same on every run.
    """
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0, random_state=0
    )
    return _fit_predict(model, train, test, cols)


def poisson_goals(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Expected goals per side by Poisson regression, H/D/A from the independent score grid."""
    rates = []
    for side in ("home", "away"):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PoissonRegressor(alpha=1e-4, max_iter=1000),
        )
        model.fit(train.select(cols).to_numpy(), train[f"{side}_goals"].to_numpy())
        rates.append(model.predict(test.select(cols).to_numpy()))
    return score_grid_proba(*rates)


def score_grid_proba(home_rate: np.ndarray, away_rate: np.ndarray) -> np.ndarray:
    """P(A), P(D), P(H) for independent Poisson goals with these means, per game."""
    goals = np.arange(MAX_GOALS + 1)
    home = np.asarray(poisson.pmf(goals, home_rate[:, None]))
    away = np.asarray(poisson.pmf(goals, away_rate[:, None]))
    grid = home[:, :, None] * away[:, None, :]  # [game, home goals, away goals]
    proba = np.stack(
        [
            np.triu(grid, 1).sum((1, 2)),
            np.trace(grid, axis1=1, axis2=2),
            np.tril(grid, -1).sum((1, 2)),
        ],
        axis=1,
    )
    return proba / proba.sum(1, keepdims=True)


MODELS: dict[str, Model] = {
    "logistic": logistic,
    "logistic + sigmoid": logistic_sigmoid,
    "boosted trees": boosted_trees,
    "poisson goals": poisson_goals,
}


def _onehot(y: np.ndarray) -> np.ndarray:
    return (y[:, None] == np.array(CLASSES)[None, :]).astype(float)


def rps(y: np.ndarray, proba: np.ndarray) -> float:
    """Mean ranked probability score, outcomes ordered H, D, A (lower is better)."""
    order = [CLASSES.index(c) for c in ("H", "D", "A")]
    gap = np.cumsum(proba[:, order], 1) - np.cumsum(_onehot(y)[:, order], 1)
    return float(np.mean(np.sum(gap[:, :-1] ** 2, 1) / (len(CLASSES) - 1)))


def scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Log loss, RPS, multiclass Brier score and accuracy of `proba` on outcomes `y`."""
    return {
        "log_loss": float(log_loss(y, proba, labels=CLASSES)),
        "rps": rps(y, proba),
        "brier": float(np.mean(np.sum((proba - _onehot(y)) ** 2, 1))),
        "accuracy": float(np.mean(np.array(CLASSES)[proba.argmax(1)] == y)),
    }


def walk_forward_predictions(
    df: pl.DataFrame, model: Model, cols: list[str], seasons: tuple[int, ...] = VALID_SEASONS
) -> tuple[np.ndarray, np.ndarray]:
    """Outcomes and predictions for `seasons`, each from a model fitted on the seasons before."""
    season = pl.col("season_start")
    ys, probas = [], []
    for s in seasons:
        test = df.filter(season == s)
        probas.append(model(df.filter(season < s), test, cols))
        ys.append(test[TARGET].to_numpy())
    return np.concatenate(ys), np.vstack(probas)


def compare(
    df: pl.DataFrame,
    models: dict[str, Model] = MODELS,
    groups: list[str] = SELECTED,
    seasons: tuple[int, ...] = VALID_SEASONS,
) -> pl.DataFrame:
    """One row per model: its scores pooled over the walk-forward seasons."""
    cols = columns(groups)
    rows = [
        {"model": n, **scores(*walk_forward_predictions(df, m, cols, seasons))}
        for n, m in models.items()
    ]
    return pl.DataFrame(rows)


def held_out(
    df: pl.DataFrame, models: dict[str, Model] = MODELS, groups: list[str] = SELECTED
) -> pl.DataFrame:
    """One row per model: its scores on the seasons from TEST_FROM, fitted on those before."""
    season, cols = pl.col("season_start"), columns(groups)
    train, test = df.filter(season < TEST_FROM), df.filter(season >= TEST_FROM)
    y = test[TARGET].to_numpy()
    return pl.DataFrame(
        [{"model": n, **scores(y, m(train, test, cols))} for n, m in models.items()]
    )


def calibration(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Per class and probability decile: games, mean predicted probability, observed rate."""
    frames = []
    for i, c in enumerate(CLASSES):
        p = proba[:, i]
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        frames.append(
            pl.DataFrame(
                {
                    "class": c,
                    "bin": np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1),
                    "predicted": p,
                    "observed": (y == c).astype(float),
                }
            )
        )
    return (
        pl.concat(frames)
        .group_by("class", "bin", maintain_order=True)
        .agg(pl.len().alias("games"), pl.col("predicted", "observed").mean())
        .sort("class", "bin")
    )


def calibration_error(table: pl.DataFrame) -> pl.DataFrame:
    """Per class: expected calibration error, the games-weighted mean |predicted - observed|."""
    return table.group_by("class", maintain_order=True).agg(
        (
            ((pl.col("predicted") - pl.col("observed")).abs() * pl.col("games")).sum()
            / pl.col("games").sum()
        ).alias("ece")
    )


@dataclass(frozen=True)
class SavedModel:
    """A fitted H/D/A model with what it needs to predict: its feature columns and classes."""

    pipeline: Any
    columns: list[str]
    classes: list[str]
    seasons: tuple[int, int]
    """First and last season_start it was fitted on."""

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        """P(A), P(D), P(H) for each row of `frame` (model_frame columns), plus its game_id."""
        x = with_derived(frame).select(pl.col(self.columns).cast(pl.Float64)).to_numpy()
        proba = self.pipeline.predict_proba(x)
        out = pl.DataFrame({f"p_{c}": proba[:, i] for i, c in enumerate(self.classes)})
        return frame.select("game_id").hstack(out) if "game_id" in frame.columns else out


def fit_final(df: pl.DataFrame, groups: list[str] = SELECTED) -> SavedModel:
    """The logistic baseline fitted on every game in `df` (load's output) with `groups`."""
    cols = columns(groups)
    pipeline = make_model().fit(df.select(cols).to_numpy(), df[TARGET].to_numpy())
    assert list(pipeline.classes_) == CLASSES
    seasons = df["season_start"].to_list()
    return SavedModel(pipeline, cols, CLASSES, (min(seasons), max(seasons)))


def save_model(model: SavedModel, path: Path) -> None:
    """Write `model` to `path` with joblib, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> SavedModel:
    """Read a model save_model wrote. joblib runs code from the file: load only your own."""
    model = joblib.load(path)
    assert isinstance(model, SavedModel), type(model)
    return model


def main(path: Path, out_file: Path) -> None:
    """Print the comparison, the baseline's calibration and the held-out scores; save the model."""
    df = load(path)
    print(f"{df.height} games, features {SELECTED}, test from {TEST_FROM}\n")
    with pl.Config(float_precision=4, tbl_rows=40):
        print("Walk-forward over", VALID_SEASONS)
        print(compare(df))
        table = calibration(*walk_forward_predictions(df, logistic, columns(SELECTED)))
        print("\nCalibration of the logistic baseline, walk-forward")
        print(calibration_error(table))
        print(table)
        print("\nHeld out:")
        print(held_out(df))
    model = fit_final(df)
    save_model(model, out_file)
    print(f"\nSaved the logistic model, fitted on seasons {model.seasons}, to {out_file}")


def exec_modeling() -> None:
    """Entry point `modeling`: compare the H/D/A models on model_frame.csv, save the chosen one."""
    main(settings.clean.out_dir / "model_frame.csv", settings.modeling.out_file)
