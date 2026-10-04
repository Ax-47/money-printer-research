"""Permutation importance of the pre-match player features.

Trains on earlier seasons, scores on the last `test_seasons`, so importance
reflects what generalises to future matches rather than what fits the past.
"""

from typing import cast

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

CLASSES = np.array(["A", "D", "H"])
TARGET = "full_time_result"


def _split(df: pl.DataFrame, first_test_season: int) -> tuple[np.ndarray, np.ndarray]:
    test = (df["season_start"] >= first_test_season).to_numpy()
    return ~test, test


def feature_importance(
    settings: Settings = default_settings,
    first_season: int = 2015,  # 2014/15 has no player history yet
    first_test_season: int = 2022,
    n_repeats: int = 20,
) -> pl.DataFrame:
    df = (
        pl.read_csv(settings.clean.out_dir / "matches.csv", try_parse_dates=True)
        .filter(pl.col("season_start") >= first_season)
        .sort("match_date")
    )
    feat_cols = [c for c in df.columns if "_player" in c]
    x = df.select(feat_cols).to_numpy()
    y = df[TARGET].to_numpy()
    train, test = _split(df, first_test_season)

    prior = np.array([(y[train] == c).mean() for c in CLASSES])
    baseline = log_loss(y[test], np.tile(prior, (test.sum(), 1)), labels=CLASSES)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    model.fit(x[train], y[train])
    proba = model.predict_proba(x[test])
    print(f"baseline log loss: {baseline:.4f}")
    print(
        f"model    log loss: {log_loss(y[test], proba, labels=CLASSES):.4f}  "
        f"acc {accuracy_score(y[test], model.predict(x[test])):.3f}"
    )

    pi = cast(
        Bunch,
        permutation_importance(
            model, x[test], y[test], scoring="neg_log_loss", n_repeats=n_repeats, random_state=0
        ),
    )
    return (
        pl.DataFrame(
            {"feature": feat_cols, "importance": pi.importances_mean, "std": pi.importances_std}
        )
        .with_columns(
            pl.col("feature").str.extract(r"^(h|a)_").alias("side"),
            pl.col("feature").str.extract(r"player(\d+)_").cast(pl.Int32).alias("slot"),
            pl.col("feature").str.extract(r"player\d+_(.+)$").alias("stat"),
        )
        .sort("importance", descending=True)
    )


def exec_feature_importance() -> None:
    imp = feature_importance()
    print(imp.head(15))
    print(imp.group_by("stat").agg(pl.col("importance").sum()).sort("importance", descending=True))
    print(imp.group_by("slot").agg(pl.col("importance").sum()).sort("slot"))
