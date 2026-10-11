import numpy as np
import polars as pl

from money_printer_research.analysis.feature_importance import (
    CLASSES,
    FEATURE_GROUPS,
    LEAGUES,
    POST_MATCH,
    SELECTED,
    columns,
    forward_selection,
    held_out,
    walk_forward,
    with_derived,
)
from money_printer_research.schema import slot_columns

SEASONS = range(2015, 2019)
GAMES = 300


def _frame(seed: int = 0) -> pl.DataFrame:
    """Games over four seasons where elo_diff drives the result and nothing else does."""
    rng = np.random.default_rng(seed)
    n = len(SEASONS) * GAMES
    elo_diff = rng.normal(0, 150, n)
    p_home = 1 / (1 + np.exp(-elo_diff / 100))
    u = rng.random(n)
    result = np.where(u < 0.75 * p_home, "H", np.where(u < 0.75 * p_home + 0.25, "D", "A"))
    noise = {
        c: rng.normal(size=n)
        for cols in FEATURE_GROUPS.values()
        for c in cols
        if c.startswith(("h_", "a_", "home_", "away_")) and not c.endswith(("_mean", "_n_low_min"))
    }
    slots = {c: rng.normal(size=n) for c in slot_columns()}
    slots["h_player3_xg90"][:50] = np.nan  # a starter under the minutes threshold
    return pl.DataFrame(
        {
            "season_start": np.repeat(list(SEASONS), GAMES),
            "league": rng.choice(LEAGUES, n),
            "full_time_result": result,
            "elo_home": 1500 + elo_diff / 2,
            "elo_away": 1500 - elo_diff / 2,
            "elo_diff": elo_diff,
            **noise,
            **slots,
        }
    ).fill_nan(None)


def test_feature_groups_never_hold_a_post_match_column() -> None:
    feats = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert not [c for c in feats if POST_MATCH.match(c)]
    assert len(feats) == len(set(feats))
    assert set(SELECTED) <= set(FEATURE_GROUPS)


def test_with_derived_builds_every_feature_column() -> None:
    df = with_derived(_frame())
    assert set(columns(list(FEATURE_GROUPS))) <= set(df.columns)
    assert df["h_n_low_min"][:50].to_list() == [1] * 50
    assert df["h_n_low_min"][50:].sum() == 0
    league_flags = df.select(columns(["league"])).sum_horizontal()
    assert league_flags.to_list() == [1] * df.height
    diff = df["form90_mean_diff"] - (df["h_form90_mean"] - df["a_form90_mean"])
    assert diff.abs().max() == 0


def test_walk_forward_scores_elo_above_noise() -> None:
    df = with_derived(_frame())
    seasons = (2017, 2018)
    assert walk_forward(df, ["elo"], seasons) < walk_forward(df, ["fatigue"], seasons)


def test_forward_selection_starts_from_elo_and_stops_on_noise() -> None:
    df = with_derived(_frame())
    steps = forward_selection(df, min_gain=0.01, seasons=(2017, 2018))
    assert steps["added"].to_list() == ["elo"]
    assert steps["gain"].to_list() == [None]


def test_held_out_beats_the_prior_with_elo() -> None:
    df = with_derived(_frame()).with_columns(
        pl.col("season_start").replace_strict({2018: 2024}, default=pl.col("season_start"))
    )
    scores = dict(held_out(df, {"elo": ["elo"]}).iter_rows())
    assert scores["elo"] < scores["class prior"]
    assert sorted(CLASSES) == CLASSES
