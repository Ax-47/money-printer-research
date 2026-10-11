import pandera.errors
import polars as pl
import pytest

from money_printer_research.features.missing import (
    feature_group,
    missing_table,
    null_report,
    reason_counts,
)
from money_printer_research.schema import (
    CLEAN_FILES,
    MatchFeaturesCols,
    MissingCols,
    N_SLOTS,
    PLAYER_SLOT_FEATURES,
    slot_column,
)


def _slot(side: str, slot: int, feature: str, game: int) -> float | None:
    """Slot values for the three games of _rows."""
    if side == "h" and game == 2:
        return None  # game 2: no home line-up
    if side == "a" and game == 3 and slot == N_SLOTS:
        return None  # game 3: only ten away starters
    if side == "h" and game == 1 and slot == 9 and feature in ("xg90", "xa90", "form90"):
        return None  # game 1: home starter 9 under the minutes threshold
    return 0.0 if feature == "prev_min" else 0.5


def _rows() -> pl.DataFrame:
    """Three games with the columns missing_table reads, in match_features form.

    1: complete but for one home starter with too few minutes, and its last game.
    2: no home line-up. 3: no Transfermarkt game, so no schedule; ten away starters.
    """
    games = [1, 2, 3]
    slots = {
        slot_column(side, s, f): pl.Series([_slot(side, s, f, g) for g in games], dtype=pl.Float64)
        for side in ("h", "a")
        for s in range(1, N_SLOTS + 1)
        for f in PLAYER_SLOT_FEATURES
    }
    return pl.DataFrame(
        {
            "game_id": games,
            "tm_game_id": ["t1", "t2", None],
            "home_formation": ["4-4-2", "4-3-3 Attacking", None],
            "home_variant": [None, "attacking", None],
            "away_formation": ["4-2-3-1", None, None],
            "away_variant": [None, None, None],
            "h_fatigue": [40.0, None, 35.0],
            "h_min_7d": [450, None, 300],
            "a_min_7d": [500, 520, 610],
            "fatigue_diff": [1.0, None, 2.0],
            "h_games_7d_all": [1, 0, None],
            "a_games_7d_all": [0, 1, None],
            "h_days_to_next_all": [None, 4, None],
            **slots,
        }
    )


def _reasons(missing: pl.DataFrame, game: int) -> dict[str, str]:
    return dict(missing.filter(pl.col("game_id") == game).select("column", "reason").rows())


def test_missing_table_gives_every_null_cell_a_reason() -> None:
    missing = missing_table(_rows()).unwrap()
    assert missing.height == _rows().null_count().sum_horizontal().item()

    assert _reasons(missing, 1) == {
        "h_days_to_next_all": "no_next_game",
        "home_variant": "plain_shape",
        "away_variant": "plain_shape",
        **{slot_column("h", 9, f): "too_few_minutes" for f in ("xg90", "xa90", "form90")},
    }

    game2 = _reasons(missing, 2)
    assert game2["h_fatigue"] == game2["h_min_7d"] == game2["fatigue_diff"] == "no_lineup"
    assert game2[slot_column("h", 1, "xg90")] == "no_lineup"
    assert game2["away_formation"] == game2["away_variant"] == "no_formation"
    assert set(game2.values()) == {"no_lineup", "no_formation"}
    assert len(game2) == 5 + N_SLOTS * len(PLAYER_SLOT_FEATURES)

    game3 = _reasons(missing, 3)
    assert game3["tm_game_id"] == game3["home_formation"] == "no_tm_game"
    assert game3["away_variant"] == "no_tm_game"
    assert game3["h_games_7d_all"] == game3["h_days_to_next_all"] == "no_schedule"
    assert game3[slot_column("a", N_SLOTS, "fatigue_score")] == "short_lineup"


def test_missing_table_sides() -> None:
    missing = missing_table(_rows()).unwrap()
    side = dict(missing.filter(pl.col("game_id") == 2).select("column", "side").rows())
    assert side["h_fatigue"] == "h"
    assert side["fatigue_diff"] is None
    assert side["away_formation"] == "a"


def test_missing_table_is_empty_without_nulls() -> None:
    complete = _rows().filter(pl.col("game_id") == 1).fill_null(0).fill_null("flat")
    assert missing_table(complete).unwrap().height == 0


def test_match_features_gains_no_columns_and_missing_is_a_clean_file() -> None:
    assert not any("has_" in c or "rated" in c for c in MatchFeaturesCols.to_schema().columns)
    assert CLEAN_FILES["missing"] is not None
    assert list(MissingCols.to_schema().columns) == ["game_id", "column", "side", "reason"]


def test_reason_counts() -> None:
    counts = reason_counts(missing_table(_rows()).unwrap())
    by_reason = {r["reason"]: r for r in counts.iter_rows(named=True)}
    assert by_reason["too_few_minutes"]["cells"] == 3
    assert by_reason["too_few_minutes"]["matches"] == 1
    assert counts["cells"].to_list() == sorted(counts["cells"].to_list(), reverse=True)


def test_feature_group_collapses_sides_and_slots() -> None:
    assert feature_group("h_player3_xg90") == "player_xg90"
    assert feature_group("a_player11_fatigue_score") == "player_fatigue_score"
    assert feature_group("a_fatigue") == "fatigue"
    assert feature_group("elo_diff") == "elo_diff"


def test_null_report_counts_per_group() -> None:
    report = null_report(_rows())
    by_group = {r["group"]: r for r in report.iter_rows(named=True)}
    assert by_group["min_7d"]["nulls"] == 1
    assert by_group["min_7d"]["columns"] == 2
    assert by_group["min_7d"]["share"] == round(1 / 6, 4)
    assert report["nulls"].to_list() == sorted(report["nulls"].to_list(), reverse=True)


def test_null_report_is_empty_without_nulls() -> None:
    assert null_report(pl.DataFrame({"h_x": [1, 2], "a_x": [3, 4]})).height == 0


def test_missing_cols_rejects_a_cell_listed_twice() -> None:
    twice = pl.DataFrame(
        {
            "game_id": [1, 1],
            "column": ["h_fatigue"] * 2,
            "side": ["h"] * 2,
            "reason": ["other"] * 2,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        MissingCols.validate(twice)
