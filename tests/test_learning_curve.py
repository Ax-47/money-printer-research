from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from money_printer_research.analysis.learning_curve import (
    held_out_games,
    learning_curve,
    references,
    render,
    spread,
    training_sizes,
)
from tests.test_modeling import _games


def _shifted() -> pl.DataFrame:
    """test_modeling's games moved to 2022-2025, so 2024 and 2025 are held out."""
    df = _games()
    start = date(2022, 8, 1)
    return df.with_columns(
        season_start=pl.col("season_start") + 7,
        game_id=pl.int_range(pl.len()),
        date=pl.Series([start + timedelta(days=i // 10) for i in range(df.height)]),
    )


def test_training_sizes_double_up_to_every_game() -> None:
    assert training_sizes(1100, 250) == [250, 500, 1000, 1100]
    assert training_sizes(250, 250) == [250]


def test_spread_keeps_labels_apart_and_leaves_far_ones_alone() -> None:
    assert spread([100, 105, 300], gap=15) == [95, 110, 300]
    assert spread([105, 100], gap=15) == [110, 95]
    assert spread([100, 101, 102], gap=10) == [91, 101, 111]


def test_the_curve_scores_every_model_on_the_same_held_out_games() -> None:
    df = _shifted()
    test = held_out_games(df, None)
    assert set(test["season_start"]) == {2024, 2025}
    curve = learning_curve(df, test, groups=["elo"], sizes=[100, 600])
    assert curve.height == 2 * 4
    assert set(curve["games"]) == {100, 600}
    # The most recent games are used: 100 of them all come from the last training season.
    assert curve.filter(pl.col("games") == 100)["first_season"].unique().to_list() == [2023]
    logistic = curve.filter(pl.col("model") == "logistic").sort("games")
    # Elo drives the result, so more games help the held-out score and close the gap.
    assert logistic["test_log_loss"][1] < logistic["test_log_loss"][0]
    gap = logistic["test_log_loss"] - logistic["train_log_loss"]
    assert gap[1] < gap[0]
    assert (curve["test_se"] > 0).all()


def test_references_score_outcome_shares_and_the_market() -> None:
    test = pl.DataFrame(
        {
            "full_time_result": ["H", "A"],
            "odds_H": [1.5, 3.0],
            "odds_D": [4.0, 3.5],
            "odds_A": [6.0, 2.2],
        }
    )
    train = pl.DataFrame({"full_time_result": ["H", "H", "D", "A"]})
    refs = references(train, test)
    assert refs["base_rates"] == pytest.approx(-(np.log(0.5) + np.log(0.25)) / 2)
    assert refs["market"] < refs["base_rates"]
    assert "market" not in references(train, test.drop("odds_H", "odds_D", "odds_A"))


def test_the_page_has_both_languages_and_no_unfilled_fields() -> None:
    df = _shifted()
    test = held_out_games(df, None)
    curve = learning_curve(df, test, groups=["elo"], sizes=[100, 600])
    pool = df.filter(pl.col("season_start") < 2024)
    page = render(curve, references(pool, test) | {"market": 0.95}, test.height, "2026-10-12")
    assert "โมเดลเรียนรู้อย่างไร" in page and "How the models learn" in page
    assert "บนนัดชุดเดียวกัน" in page  # the market finding
    assert page.count("<svg") == 2
    body = page.split("</style>")[1].split("<script>")[0]
    assert "{" not in body
