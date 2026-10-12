"""Feature selection for the H/D/A model on model_frame.csv.

Features come from an explicit allowlist of groups (FEATURE_GROUPS), each built from a
pre-match source; nothing outside it can reach the model, so the result of the match
(goals, xG, is_result) cannot leak in. A group is one feature for both sides plus their
difference, so a side is never kept without the other.

Selection is greedy and forward, by group, starting from Elo: add the group that lowers
the walk-forward log loss most, until no group lowers it by MIN_GAIN. Walk-forward means
each season in VALID_SEASONS is scored by a model fitted on the seasons before it only.
The finished seasons from TEST_FROM on are touched once, at the end, to compare the
selection with Elo alone and with every group. A season still being played is left out
of that test (held_out_split): its games are a few weeks of one part of the calendar, and
the newest ones have no Transfermarkt schedule yet.

The model is a multinomial logistic regression on standardised features, nulls filled
with the training median: on this data it beats gradient-boosted trees, which overfit
the few hundred games a feature's signal rests on.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from money_printer_research.config import settings
from money_printer_research.schema import PLAYER_SLOT_FEATURES, read_clean

TARGET = "full_time_result"
CLASSES = ["A", "D", "H"]
"""Sorted, the order of predict_proba's columns."""

TEST_FROM = 2024
VALID_SEASONS = (2021, 2022, 2023)
MIN_GAIN = 0.0005
"""Smallest drop in mean walk-forward log loss for which a group is added."""

UNFINISHED_SHARE = 0.5
"""The newest season counts as still being played below this share of the games before it."""

POST_MATCH = re.compile(
    r"^(full_time_.*|half_time_.*|(home|away)_(goals|xg)|tm_(home|away)_goals|is_result)$"
)
"""Known only after the match. No feature group may contain one (checked in load)."""

SCHEDULE = (
    "rest_days_all",
    "games_7d_all",
    "games_14d_all",
    "non_league_14d",
    "prev_europe_away",
    "days_to_next_all",
    "next_is_europe",
)


LEAGUES = tuple(sorted(settings.understat.leagues))


def _league_column(league: str) -> str:
    """A 0/1 column's name: "ESP-La Liga" -> "league_esp_la_liga"."""
    return "league_" + re.sub(r"\W+", "_", league).lower()


def _sides(name: str, diff: bool = True) -> list[str]:
    """h_<name>, a_<name> and, if diff, <name>_diff (see with_derived)."""
    return [f"h_{name}", f"a_{name}", *([f"{name}_diff"] if diff else [])]


FEATURE_GROUPS: dict[str, list[str]] = {
    "elo": ["elo_home", "elo_away", "elo_diff"],
    "league": [_league_column(lg) for lg in LEAGUES],
    "fatigue": _sides("fatigue"),
    "min_7d": _sides("min_7d"),
    **{s: _sides(s, diff=s not in ("prev_europe_away", "next_is_europe")) for s in SCHEDULE},
    "formation": [
        f"{p}_{x}"
        for p in ("home", "away")
        for x in ("defenders", "midfielders", "forwards", "lines")
    ],
    "n_low_min": _sides("n_low_min", diff=False),
    **{f"player_{f}": _sides(f"{f}_mean") for f in PLAYER_SLOT_FEATURES},
}
"""Every candidate feature, by group. Player slots enter as each side's starters' mean."""

SELECTED = ["elo", "player_form90", "games_14d_all", "player_prev_min"]
"""Groups forward selection kept on the real data (2026-10-11): walk-forward log loss
0.9842 for Elo alone, 0.9811 with these; on 2024-2025, 0.9893 and 0.9855 (every group:
0.9877). Rerun `uv run feature_importance` after the data changes."""


def with_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Add the derived columns FEATURE_GROUPS names: slot means, diffs, league flags.

    <f>_mean is the mean of a side's starter slots (nulls skipped), n_low_min the number
    of its starters under MIN_PREV_MIN previous minutes, <name>_diff home minus away.
    A diff whose h_/a_ columns are not in `df` is left out (predict's fixtures carry only
    the features the saved model uses); selecting it later fails loudly.
    """
    slots = {
        (s, f): [c for c in df.columns if re.fullmatch(rf"{s}_player\d+_{f}", c)]
        for s in ("h", "a")
        for f in PLAYER_SLOT_FEATURES
    }
    df = df.with_columns(
        *((pl.col("league") == lg).cast(pl.Int8).alias(_league_column(lg)) for lg in LEAGUES),
        *(pl.mean_horizontal(cols).alias(f"{s}_{f}_mean") for (s, f), cols in slots.items()),
        *(
            pl.sum_horizontal(pl.col(c).is_null() for c in slots[s, "xg90"]).alias(
                f"{s}_n_low_min"
            )
            for s in ("h", "a")
        ),
    )
    diffs = {
        c.removesuffix("_diff")
        for cols in FEATURE_GROUPS.values()
        for c in cols
        if c.endswith("_diff")
    }
    present = [n for n in sorted(diffs - {"elo"}) if {f"h_{n}", f"a_{n}"} <= set(df.columns)]
    return df.with_columns(
        (pl.col(f"h_{n}") - pl.col(f"a_{n}")).alias(f"{n}_diff") for n in present
    )


def load(path: Path) -> pl.DataFrame:
    """model_frame.csv with the derived columns, every feature as Float64."""
    df = with_derived(read_clean(path.parent, path.stem))
    feats = [c for cols in FEATURE_GROUPS.values() for c in cols]
    leaking = [c for c in feats if POST_MATCH.match(c)]
    assert not leaking, leaking
    return df.with_columns(pl.col(feats).cast(pl.Float64))


def columns(groups: list[str]) -> list[str]:
    """The feature columns of `groups`, in order."""
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def make_model() -> Pipeline:
    """Median fill, standardise, multinomial logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
    )


def fit_score(train: pl.DataFrame, test: pl.DataFrame, groups: list[str]) -> float:
    """Log loss on `test` of make_model fitted on `train` with the columns of `groups`."""
    cols = columns(groups)
    model = make_model().fit(train.select(cols).to_numpy(), train[TARGET].to_numpy())
    proba = model.predict_proba(test.select(cols).to_numpy())
    return float(log_loss(test[TARGET].to_numpy(), proba, labels=CLASSES))


def prior_score(train: pl.DataFrame, test: pl.DataFrame) -> float:
    """Log loss of predicting every game with the training shares of A, D and H."""
    shares = [float((train[TARGET] == c).mean()) for c in CLASSES]  # pyright: ignore[reportArgumentType]
    return float(log_loss(test[TARGET].to_numpy(), [shares] * test.height, labels=CLASSES))


def walk_forward(
    df: pl.DataFrame, groups: list[str], seasons: tuple[int, ...] = VALID_SEASONS
) -> float:
    """Mean log loss over `seasons`, each scored by a model fitted on the seasons before it."""
    season = pl.col("season_start")
    return float(
        np.mean(
            [fit_score(df.filter(season < s), df.filter(season == s), groups) for s in seasons]
        )
    )


def forward_selection(
    df: pl.DataFrame,
    start: list[str] | None = None,
    min_gain: float = MIN_GAIN,
    seasons: tuple[int, ...] = VALID_SEASONS,
) -> pl.DataFrame:
    """Greedy forward selection by group; one row per step: the group added and the loss."""
    chosen = ["elo"] if start is None else list(start)
    best = walk_forward(df, chosen, seasons)
    steps: list[dict[str, Any]] = [{"added": "+".join(chosen), "log_loss": best, "gain": None}]
    while rest := [g for g in FEATURE_GROUPS if g not in chosen]:
        scores = {g: walk_forward(df, [*chosen, g], seasons) for g in rest}
        group = min(scores, key=scores.__getitem__)
        if best - scores[group] < min_gain:
            break
        chosen.append(group)
        steps.append({"added": group, "log_loss": scores[group], "gain": best - scores[group]})
        best = scores[group]
    return pl.DataFrame(
        steps, schema={"added": pl.String, "log_loss": pl.Float64, "gain": pl.Float64}
    )


def last_finished_season(df: pl.DataFrame) -> int:
    """The newest season in df, or the one before it when the newest is still being played.

    A season is taken as still being played when it has fewer than UNFINISHED_SHARE of the
    games of the season before it; a league shrinking (Ligue 1 from 20 clubs to 18) does not
    come near that.
    """
    counts = df.group_by("season_start").len().sort("season_start")
    seasons, games = counts["season_start"].to_list(), counts["len"].to_list()
    if len(seasons) > 1 and games[-1] < UNFINISHED_SHARE * games[-2]:
        return seasons[-2]
    return seasons[-1]


def held_out_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(train, test): the seasons before TEST_FROM, and the finished seasons from it on."""
    season, last = pl.col("season_start"), last_finished_season(df)
    return df.filter(season < TEST_FROM), df.filter(season.is_between(TEST_FROM, last))


def held_out(df: pl.DataFrame, candidates: dict[str, list[str]]) -> pl.DataFrame:
    """Log loss of each candidate on the finished seasons from TEST_FROM, fitted before them."""
    train, test = held_out_split(df)
    rows = [{"features": "class prior", "log_loss": prior_score(train, test)}]
    rows += [{"features": n, "log_loss": fit_score(train, test, g)} for n, g in candidates.items()]
    return pl.DataFrame(rows)


def main(path: Path) -> None:
    """Print the forward selection, then the held-out comparison."""
    df = load(path)
    last = last_finished_season(df)
    print(
        f"{df.height} games, {len(FEATURE_GROUPS)} feature groups,"
        f" test {TEST_FROM} to {last} (finished seasons)\n"
    )
    steps = forward_selection(df.filter(pl.col("season_start") < TEST_FROM))
    chosen = [g for s in steps["added"] for g in s.split("+")]
    with pl.Config(float_precision=4):
        print("Forward selection, walk-forward log loss over", VALID_SEASONS)
        print(steps)
        print("\nHeld out:")
        print(
            held_out(df, {"elo": ["elo"], "selected": chosen, "every group": list(FEATURE_GROUPS)})
        )
    if chosen != SELECTED:
        print(f"\nselection differs from SELECTED: {chosen}")


def exec_feature_importance() -> None:
    """Entry point `feature_importance`: forward selection on model_frame.csv."""
    main(settings.clean.out_dir / "model_frame.csv")
