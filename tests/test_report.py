"""analysis.report: the page is built from the results it is given."""

import datetime

import polars as pl

from money_printer_research.analysis.report import interval_chart, render

DAY = datetime.date(2024, 8, 17)


def test_interval_chart_has_a_row_and_tooltip_per_estimate() -> None:
    svg = interval_chart([("a", 0.01, -0.01, 0.03), ("b <c>", -0.02, -0.03, -0.01)], "test")
    assert svg.count("<g class=") == 2
    assert "b &lt;c&gt;: -2.0% (95% interval -3.0% to -1.0%)" in svg
    assert 'class="row above"' not in svg  # neither interval is wholly above zero


def _results() -> dict:
    variant = "value 1X2, best price early [min_edge=0.02]"
    comparisons = pl.DataFrame(
        {
            "a": ["close", "market_and_form"],
            "b": ["early", "early"],
            "games": [100, 100],
            "log_loss": [-0.0024, 0.0008],
            "log_loss_low": [-0.0035, -0.0001],
            "log_loss_high": [-0.0013, 0.0017],
            "rps": [0.0, 0.0],
            "rps_low": [0.0, 0.0],
            "rps_high": [0.0, 0.0],
        }
    )
    scores = pl.DataFrame(
        {
            "strategy": ["always home", "value 1X2, best price early"],
            "variant": ["always home", variant],
            "league": ["ALL", "ALL"],
            "bets": [100, 50],
            "profit": [-4.0, 1.0],
            "roi": [-0.04, 0.02],
            "roi_low": [-0.06, -0.02],
            "roi_high": [-0.02, 0.06],
            "close_ev": [-0.03, 0.03],
            "close_ev_low": [-0.031, 0.027],
            "close_ev_high": [-0.029, 0.033],
            "p_value": [0.9, 0.2],
            "q_value": [1.0, 0.4],
        }
    )
    return {
        "generated": "2026-10-12",
        "code_commit": "abc1234",
        "dirty": False,
        "suite_commit": "def5678 2026-10-12",
        "games": pl.DataFrame({"season_start": [2014, 2025], "league": ["L", "L"]}),
        "test_games": 100,
        "margins": pl.DataFrame({"method": ["power"], "timing": ["close"], "log_loss": [0.96]}),
        "forecasters": pl.DataFrame(
            {"forecaster": ["close", "early"], "log_loss": [0.96, 0.963], "rps": [0.19, 0.194]}
        ),
        "comparisons": comparisons,
        "by_league": comparisons.tail(1).with_columns(league=pl.lit("L")),
        "scores": scores,
        "walk_forward": pl.DataFrame(
            {
                "strategy": ["always home"],
                "league": ["ALL"],
                "seasons": [9],
                "bets": [90],
                "profit": [-4.0],
                "roi": [-0.04],
                "roi_low": [-0.06],
                "roi_high": [-0.02],
            }
        ),
        "test_from": 2016,
        "staking": {
            "replay": pl.DataFrame(
                {
                    "variant": [variant],
                    "plan": ["flat"],
                    "final": [2.4],
                    "max_drawdown": [0.4],
                    "lowest": [0.65],
                }
            ),
            "ruin": pl.DataFrame(
                {
                    "variant": [variant, variant],
                    "plan": ["martingale", "martingale"],
                    "seasons": [1, 3],
                    "p_lose_half": [0.97, 0.99],
                    "p_lose_90": [0.96, 0.99],
                    "median_final": [0.0, 0.0],
                }
            ),
        },
        "time": pl.DataFrame(
            {
                "variant": [variant],
                "bets_per_season": [624],
                "close_ev": [0.03],
                "units_per_season_close_ev": [19.0],
                "seasons_needed_close_ev": [23.0],
            }
        ),
        "limits": pl.DataFrame(
            {
                "accounts": [None, 5],
                "limit": [None, 5.0],
                "bets": [6240, 1108],
                "profit": [143.0, 32.0],
                "last_day": [DAY, DAY],
            }
        ),
        "exchange": pl.DataFrame(
            {"commission": [0.02], "bets": [318], "roi": [-0.18], "close_ev": [0.0]}
        ),
    }


def test_render_states_the_results_it_was_given() -> None:
    page = render(_results())
    assert page.startswith("<!doctype html>")
    assert "code abc1234 ·" in page
    assert "2014/15 to 2025/26" in page
    assert "+0.0008 (95% interval -0.0001 to +0.0017" in page
    assert "Of 2 strategy × league cells, 0 have" in page
    assert "takes 23 to 23 seasons" in page
    assert "returns -18.0% after 2% commission" in page
    assert "in 97% to 97% of simulated histories" in page
    assert "{" not in page.split("<style>")[1].split("</style>")[1]  # no unfilled fields


def test_to_json_is_plain_data_for_the_api() -> None:
    import json

    from money_printer_research.analysis.report import to_json

    data = json.loads(json.dumps(to_json(_results())))
    assert data["meta"]["first_season"] == 2014
    assert data["summary"][0]["headline"] == "No model built here forecasts better than the market."
    assert data["practicality"]["limits"][0]["last_day"] == "2024-08-17"
    assert {r["league"] for r in data["strategies"]["scores"]} == {"ALL"}
