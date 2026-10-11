import datetime

import polars as pl

from money_printer_research.clean_data.integrity import changed_rows, compare, disagree


def _before() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "goals": [1, 2, 0, 3],
            "xg": [1.5, 0.25, None, 2.0],
            "date": [datetime.date(2024, 8, d) for d in (1, 2, 3, 4)],
        }
    )


def test_compare_finds_nothing_when_a_step_only_adds_columns() -> None:
    after = _before().with_columns(pl.lit(7).alias("feature"))
    report = compare(_before(), after, "game_id", step="add").row(0, named=True)
    assert report == {
        "step": "add",
        "rows_before": 4,
        "rows_after": 4,
        "lost_keys": 0,
        "new_keys": 0,
        "dup_keys": 0,
        "changed": 0,
        "new_nulls": 0,
    }


def test_compare_counts_every_kind_of_damage() -> None:
    after = pl.DataFrame(
        {
            # 4 lost, 5 new, 2 duplicated
            "game_id": [1, 2, 2, 3, 5],
            # game 1 changed; game 2 kept in both copies
            "goals": [9, 2, 2, 0, 1],
            # game 1 became null; game 3 was null before, so not a new null
            "xg": [None, 0.25, 0.25, 1.0, 0.0],
            "date": [datetime.date(2024, 8, d) for d in (1, 2, 2, 3, 5)],
        }
    )
    report = compare(_before(), after, "game_id").row(0, named=True)
    assert report["lost_keys"] == 1
    assert report["new_keys"] == 1
    assert report["dup_keys"] == 1
    assert report["changed"] == 1
    assert report["new_nulls"] == 1


def test_compare_only_checks_the_given_columns() -> None:
    after = _before().with_columns(pl.col("goals") + 1)
    assert compare(_before(), after, "game_id", cols=["xg"])["changed"].item() == 0
    assert compare(_before(), after, "game_id")["changed"].item() == 4


def test_compare_catches_values_lost_in_a_csv_round_trip(tmp_path) -> None:
    path = tmp_path / "t.csv"
    _before().with_columns(pl.lit("not a date").alias("date")).write_csv(path)
    report = compare(_before(), pl.read_csv(path), "game_id").row(0, named=True)
    assert report["new_nulls"] == 4
    assert compare(_before(), _before(), "game_id")["new_nulls"].item() == 0


def test_compare_takes_a_compound_key() -> None:
    before = pl.DataFrame({"game_id": [1, 1], "player_id": [7, 8], "minutes": [90, 45]})
    after = before.with_columns(pl.Series("minutes", [90, 46]))
    report = compare(before, after, ["game_id", "player_id"]).row(0, named=True)
    assert (report["dup_keys"], report["changed"]) == (0, 1)


def test_changed_rows_lists_the_cells_behind_the_counts() -> None:
    after = _before().with_columns(
        pl.Series("goals", [1, 5, 0, 3]), pl.Series("xg", [1.5, 0.25, None, None])
    )
    rows = changed_rows(_before(), after, "game_id").sort("game_id")
    assert rows.rows() == [(2, "goals", "2", "5"), (4, "xg", "2.0", None)]


def test_changed_rows_is_empty_but_typed_without_changes() -> None:
    rows = changed_rows(_before(), _before(), "game_id")
    assert rows.height == 0
    assert rows.columns == ["game_id", "column", "before", "after"]


def test_disagree_keeps_rows_where_two_sources_differ() -> None:
    df = pl.DataFrame(
        {"home_goals": [1, 2, 3, 0], "tm_home_goals": [1, 3, None, 0], "game_id": [1, 2, 3, 4]}
    )
    assert disagree(df, {"home_goals": "tm_home_goals"})["game_id"].to_list() == [2]
