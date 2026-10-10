"""Feature importance for the H/D/A model on data/clean/matches.csv.

Train on seasons before TEST_FROM, score on the rest (no shuffling across time).
Importance = rise in test log loss when a feature (or a group of features) is
permuted. Groups matter here: the 22 player slots per feature are strongly
correlated, so per-column importance spreads the signal thin.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from money_printer_research.config import settings

TEST_FROM = 2024
N_REPEATS = 5
SEED = 0
TARGET = "full_time_result"
CLASSES = ["A", "D", "H"]

# Known only after the match: using any of these as a feature leaks the result.
POST_MATCH = re.compile(
    r"^(full_time_.*|half_time_.*|home_xg|away_xg"
    r"|(home|away)_(shots|shots_on_target|corners|fouls|yellow_cards|red_cards))$"
)
NOT_FEATURES = {
    "league",
    "season",
    "season_start",
    "game_id",
    "match_date",
    "home_team",
    "away_team",
}


def feature_group(col: str) -> str:
    """Collapse h/a sides and player slots: h_player3_xg90 -> player_xg90."""
    if m := re.fullmatch(r"[ha]_player\d+_(.+)", col):
        return f"player_{m[1]}"
    if col.startswith("elo"):
        return "elo"
    if col == "league_code":
        return "league"
    return re.sub(r"^[ha]_", "", col)


def load(path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Read the matches table; return it with every feature cast to float, and the features."""
    df = pl.read_csv(path, infer_schema_length=None).with_columns(
        pl.col("league").cast(pl.Categorical).to_physical().alias("league_code")
    )
    feats = [c for c in df.columns if c not in NOT_FEATURES and not POST_MATCH.match(c)]
    df = df.with_columns(pl.col(feats).cast(pl.Float64))
    return df, feats


def split_by_season(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(train, test): seasons before TEST_FROM, and the rest."""
    return (
        df.filter(pl.col("season_start") < TEST_FROM),
        df.filter(pl.col("season_start") >= TEST_FROM),
    )


def fit_model(x: np.ndarray, y: np.ndarray, feats: list[str]) -> HistGradientBoostingClassifier:
    """Gradient-boosted H/D/A classifier, with the league code as its categorical feature."""
    categorical: Any = [feats.index("league_code")]
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        categorical_features=categorical,
        random_state=SEED,
    ).fit(x, y)


def column_groups(feats: list[str]) -> dict[str, list[int]]:
    """Feature group (see feature_group) -> indices of its columns in `feats`."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(feats):
        groups.setdefault(feature_group(c), []).append(i)
    return groups


def permutation_loss(model, x: np.ndarray, y: np.ndarray, cols: list[int], rng) -> float:
    """Log loss after shuffling the rows of `cols`, all with the same permutation."""
    x_perm = x.copy()
    idx = rng.permutation(len(x))
    x_perm[:, cols] = x[idx][:, cols]  # same permutation for the whole group
    return float(log_loss(y, model.predict_proba(x_perm), labels=CLASSES))


def permutation_importance(
    model, x: np.ndarray, y: np.ndarray, groups: dict[str, list[int]], base: float, rng
) -> pl.DataFrame:
    """Mean and std rise in log loss over N_REPEATS shuffles of each group, largest first."""
    rows = []
    for name, cols in groups.items():
        d = [permutation_loss(model, x, y, cols, rng) - base for _ in range(N_REPEATS)]
        rows.append((name, len(cols), float(np.mean(d)), float(np.std(d))))
    return pl.DataFrame(
        rows, schema=["feature", "n_cols", "loss_increase", "std"], orient="row"
    ).sort("loss_increase", descending=True)


def main(path: Path) -> None:
    """Fit on the training seasons, then print test log loss and permutation importance."""
    df, feats = load(path)
    train, test = split_by_season(df)
    x_tr, y_tr = train.select(feats).to_numpy(), train[TARGET].to_numpy()
    x_te, y_te = test.select(feats).to_numpy(), test[TARGET].to_numpy()
    model = fit_model(x_tr, y_tr, feats)

    prior = train[TARGET].value_counts(normalize=True).sort(TARGET)["proportion"].to_numpy()
    base = float(log_loss(y_te, model.predict_proba(x_te), labels=CLASSES))
    naive = float(log_loss(y_te, np.tile(prior, (len(y_te), 1)), labels=CLASSES))
    print(f"train {train.height} rows, test {test.height} rows, {len(feats)} features")
    print(f"test log loss: model {base:.4f} | class prior {naive:.4f} | gain {naive - base:.4f}\n")

    rng = np.random.default_rng(SEED)
    single_columns = {c: [i] for i, c in enumerate(feats)}
    with pl.Config(tbl_rows=60, float_precision=5):
        print("Grouped permutation importance (higher = more useful):")
        print(permutation_importance(model, x_te, y_te, column_groups(feats), base, rng))
        print("\nTop 20 single columns:")
        print(permutation_importance(model, x_te, y_te, single_columns, base, rng).head(20))


def exec_feature_importance() -> None:
    """Entry point: feature importance on the cleaned matches table."""
    main(settings.clean.out_dir / "matches.csv")
