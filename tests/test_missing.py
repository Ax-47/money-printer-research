import polars as pl

from money_printer_research.features.missing import (
    FLAG_COLUMNS,
    add_missing_flags,
    feature_group,
    null_report,
)
from money_printer_research.schema import MatchFeaturesCols, N_SLOTS, slot_column


def _rows() -> pl.DataFrame:
    """Three games: complete; home line-up missing; away club without a Transfermarkt game.

    Only the columns add_missing_flags reads. In the first game, home starters 1 to 8 have
    per-90 rates and 9 to 11 are below the minutes threshold.
    """
    home_form = [
        [0.1 if s <= 8 else None for s in range(1, N_SLOTS + 1)],
        [None] * N_SLOTS,
        [0.2] * N_SLOTS,
    ]
    away_form = [[0.3] * N_SLOTS, [0.3] * N_SLOTS, [0.3] * N_SLOTS]
    slots = {
        slot_column(side, s, "form90"): pl.Series([g[s - 1] for g in form], dtype=pl.Float64)
        for side, form in (("h", home_form), ("a", away_form))
        for s in range(1, N_SLOTS + 1)
    }
    return pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "h_min_7d": [450, None, 300],
            "a_min_7d": [500, 520, 610],
            "h_games_7d_all": [1, 0, 2],
            "a_games_7d_all": [0, 1, None],
            **slots,
        }
    )


def test_add_missing_flags_names_the_missing_source() -> None:
    out = add_missing_flags(_rows())
    assert out["h_has_lineup"].to_list() == [True, False, True]
    assert out["a_has_lineup"].to_list() == [True, True, True]
    assert out["h_has_schedule"].to_list() == [True, True, True]
    assert out["a_has_schedule"].to_list() == [True, True, False]
    assert out["h_rated_starters"].to_list() == [8, 0, 11]
    assert out["a_rated_starters"].to_list() == [11, 11, 11]
    assert out.select(FLAG_COLUMNS).null_count().sum_horizontal().item() == 0


def test_add_missing_flags_leaves_the_features_alone() -> None:
    rows = _rows()
    assert add_missing_flags(rows).select(rows.columns).equals(rows)


def test_every_flag_is_in_the_match_features_schema() -> None:
    assert set(FLAG_COLUMNS) <= set(MatchFeaturesCols.to_schema().columns)


def test_feature_group_collapses_sides_and_slots() -> None:
    assert feature_group("h_player3_xg90") == "player_xg90"
    assert feature_group("a_player11_fatigue_score") == "player_fatigue_score"
    assert feature_group("a_fatigue") == "fatigue"
    assert feature_group("elo_diff") == "elo_diff"


def test_null_report_counts_per_group() -> None:
    report = null_report(_rows())
    by_group = {r["group"]: r for r in report.iter_rows(named=True)}
    assert set(by_group) == {"player_form90", "min_7d", "games_7d_all"}
    assert by_group["player_form90"]["nulls"] == 3 + N_SLOTS
    assert by_group["player_form90"]["columns"] == 2 * N_SLOTS
    assert by_group["min_7d"]["share"] == round(1 / 6, 4)
    assert report["nulls"].to_list() == sorted(report["nulls"].to_list(), reverse=True)


def test_null_report_is_empty_without_nulls() -> None:
    assert null_report(pl.DataFrame({"h_x": [1, 2], "a_x": [3, 4]})).height == 0
