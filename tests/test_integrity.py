import datetime
from typing import cast

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data.integrity import (
    at_morphism,
    changed_rows,
    compare,
    disagree,
    log_report,
    trace,
)
from money_printer_research.clean_data.match_stat import MATCH_STAT_STEPS, clean_match_stat
from money_printer_research.cool_stuff import told
from money_printer_research.schema.match.raw import UnderstatScheduleRaw


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


def test_at_morphism_logs_missing_data_at_each_step() -> None:
    def drop_last(df: pl.DataFrame) -> pl.DataFrame:
        return df.head(3)

    def blank_goals(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.when(pl.col("game_id") == 1).then(pl.col("goals")))

    pipeline = at_morphism(drop_last, key="game_id") >> at_morphism(blank_goals, key="game_id")
    logged = pipeline(told(_before())).unwrap()
    assert logged.value.height == 3
    report = log_report(logged)
    assert report["step"].to_list() == ["drop_last", "blank_goals"]
    assert report["lost_keys"].to_list() == [1, 0]
    assert report["new_nulls"].to_list() == [0, 2]
    assert report["nulls_in"].to_list() == [1, 1]
    assert report["nulls_out"].to_list() == [1, 3]
    assert report["more_nulls"].to_list() == [[], ["goals"]]


def test_at_morphism_without_a_key_or_a_frame_input() -> None:
    def to_frame(n: int) -> pl.DataFrame:
        return pl.DataFrame({"x": [None] * n}, schema={"x": pl.Int64})

    entry = log_report(at_morphism(to_frame)(told(2)).unwrap()).row(0, named=True)
    assert entry["rows_in"] is None
    assert entry["rows_out"] == 2
    assert entry["more_nulls"] == ["x"]
    assert entry["lost_keys"] is None


def test_at_morphism_reads_the_input_frame_from_source() -> None:
    def keep_first(pair: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        return pair[1]

    step = at_morphism(keep_first, key="game_id", source=lambda pair: pair[1])
    entry = log_report(step(told((_before().head(1), _before()))).unwrap()).row(0, named=True)
    assert (entry["rows_in"], entry["lost_keys"], entry["changed"]) == (4, 0, 0)


def test_trace_logs_every_step() -> None:
    logged = trace(lambda df: df, lambda df: df.head(2), key="game_id")(told(_before()))
    assert log_report(logged.unwrap())["rows_out"].to_list() == [4, 2]


def _raw_schedule() -> pl.DataFrame:
    """Three Premier League fixtures in UnderstatScheduleRaw form; the last was never played."""
    n = 3
    played = [True, True, False]
    return pl.DataFrame(
        {
            "league": ["ENG-Premier League"] * n,
            "season": [2324] * n,
            "game": ["2023-08-12 Arsenal-Chelsea"] * n,
            "league_id": [1] * n,
            "season_id": [2023] * n,
            "game_id": [101, 102, 103],
            "date": [datetime.datetime(2023, 8, d, 15) for d in (12, 19, 26)],
            "home_team_id": [83, 80, 83],
            "away_team_id": [80, 83, 80],
            "home_team": ["Arsenal", "Chelsea", "Arsenal"],
            "away_team": ["Chelsea", "Arsenal", "Chelsea"],
            "away_team_code": ["CHE", "ARS", "CHE"],
            "home_team_code": ["ARS", "CHE", "ARS"],
            "home_goals": [2, 1, None],
            "away_goals": [0, 1, None],
            "home_xg": [1.5, 0.8, None],
            "away_xg": [0.4, 1.1, None],
            "is_result": played,
            "has_data": played,
            "url": [f"https://understat.com/match/{g}" for g in (101, 102, 103)],
        }
    )


def test_trace_logs_each_morphism_of_clean_match_stat() -> None:
    raw = cast(DataFrame[UnderstatScheduleRaw], _raw_schedule())
    logged = trace(*MATCH_STAT_STEPS, key="game_id")(told(raw)).unwrap()
    assert logged.value.equals(clean_match_stat(raw).unwrap())

    report = log_report(logged)
    assert report.height == len(MATCH_STAT_STEPS)
    # Only the first morphism drops a row (the unplayed fixture); none adds a null.
    assert report["lost_keys"].to_list() == [1, 0, 0, 0, 0, 0]
    assert report["new_nulls"].to_list() == [0] * len(MATCH_STAT_STEPS)
    assert report["nulls_out"].to_list() == [0] * len(MATCH_STAT_STEPS)
    # norm_team_names rewrites both names on every row, on purpose.
    assert report.filter(pl.col("step") == "norm_team_names")["changed"].item() == 4
