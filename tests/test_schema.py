"""The schema package: its custom checks, the raw -> selected map, and the clean CSV files."""

import datetime
from dataclasses import fields
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest
from pandera.errors import SchemaErrors

import money_printer_research.schema as schema
from money_printer_research.clean_data import Cleaned, ExtractedFeatures
from money_printer_research.schema import (
    CLEAN_FILES,
    N_SLOTS,
    PLAYER_SLOT_FEATURES,
    PLAYER_SLOTS,
    SELECTED,
    EloCols,
    FormationCols,
    MatchFeaturesCols,
    MatchLeaguesCols,
    PredictionCols,
    dtype,
    match_features_schema,
    read_clean,
    slot_column,
    slot_columns,
    validate_lazy,
)


def _failed_checks(df: pl.DataFrame, model: type[pa.DataFrameModel]) -> set[str]:
    """Names of the checks `df` fails; the ValueError message wraps long names."""
    try:
        model.validate(df, lazy=True)
    except SchemaErrors as e:
        return set(e.failure_cases["check"].to_list())
    return set()


def _elo(elo_diff: float) -> pl.DataFrame:
    """One valid EloCols row whose elo_diff is given."""
    return pl.DataFrame(
        {
            "season": ["2014/15"],
            "match_date": [datetime.date(2014, 8, 16)],
            "home_team": ["arsenal"],
            "away_team": ["crystal palace"],
            "full_time_home_goals": [2],
            "full_time_away_goals": [1],
            "full_time_result": ["H"],
            "elo_home": [1500.0],
            "elo_away": [1450.0],
            "elo_diff": [elo_diff],
        }
    )


def test_elo_accepts_diff_of_home_minus_away() -> None:
    validate_lazy(_elo(50.0), EloCols.to_schema(), "elo")


def test_elo_rejects_diff_that_is_not_home_minus_away() -> None:
    assert _failed_checks(_elo(-50.0), EloCols) == {"diff_is_home_minus_away"}


def test_elo_rejects_unknown_result() -> None:
    with pytest.raises(ValueError, match="full_time_result"):
        validate_lazy(
            _elo(50.0).with_columns(full_time_result=pl.lit("W")), EloCols.to_schema(), "elo"
        )


def _predictions(p_home: float, p_draw: float, p_away: float) -> pl.DataFrame:
    return _elo(50.0).with_columns(
        lam_home=pl.lit(1.5),
        lam_away=pl.lit(1.1),
        p_home=pl.lit(p_home),
        p_draw=pl.lit(p_draw),
        p_away=pl.lit(p_away),
        rho=pl.lit(0.0),
    )


def test_predictions_accept_probabilities_that_sum_to_one() -> None:
    validate_lazy(_predictions(0.5, 0.3, 0.2), PredictionCols.to_schema(), "predictions")


def test_predictions_reject_probabilities_that_do_not_sum_to_one() -> None:
    failed = _failed_checks(_predictions(0.5, 0.3, 0.3), PredictionCols)
    assert failed == {"probabilities_sum_to_one"}


def _formation(home_shape: str | None, away_shape: str | None) -> pl.DataFrame:
    """Shapes with the line counts left null: only the shape checks are under test."""
    int8_nulls = pl.Series([None], dtype=pl.Int8)
    return pl.DataFrame(
        {
            "home_shape": pl.Series([home_shape], dtype=pl.String),
            "home_variant": pl.Series([None], dtype=pl.String),
            "home_defenders": int8_nulls,
            "home_midfielders": int8_nulls,
            "home_forwards": int8_nulls,
            "home_lines": int8_nulls,
            "away_shape": pl.Series([away_shape], dtype=pl.String),
            "away_variant": pl.Series([None], dtype=pl.String),
            "away_defenders": int8_nulls,
            "away_midfielders": int8_nulls,
            "away_forwards": int8_nulls,
            "away_lines": int8_nulls,
        }
    )


@pytest.mark.parametrize(("home", "away"), [("4-2-3-1", "3-5-2"), ("4-4-2", None), (None, None)])
def test_formation_accepts_ten_outfield_players_or_no_lineup(
    home: str | None, away: str | None
) -> None:
    validate_lazy(_formation(home, away), FormationCols.to_schema(), "formation")


def test_formation_rejects_lines_that_do_not_sum_to_ten() -> None:
    failed = _failed_checks(_formation("4-4-3", "4-4-2"), FormationCols)
    assert failed == {"lines_hold_ten_outfield_players"}


def test_formation_rejects_a_shape_that_is_not_digits_and_dashes() -> None:
    with pytest.raises(ValueError, match="home_shape"):
        validate_lazy(_formation("4-3-3 Attacking", None), FormationCols.to_schema(), "formation")


def test_formation_rejects_line_counts_out_of_range() -> None:
    frame = _formation("4-4-2", None).with_columns(home_defenders=pl.lit(6, dtype=pl.Int8))
    with pytest.raises(ValueError, match="home_defenders"):
        validate_lazy(frame, FormationCols.to_schema(), "formation")


@pytest.mark.parametrize("raw", list(SELECTED), ids=lambda m: m.__name__)
def test_selected_columns_are_copied_from_raw(raw: type[pa.DataFrameModel]) -> None:
    """Each selected column has the raw column's dtype, nullability and description."""
    raw_cols = raw.to_schema().columns
    selected_cols = SELECTED[raw].to_schema().columns
    assert selected_cols, f"{SELECTED[raw].__name__} selects no columns"
    for name, col in selected_cols.items():
        assert name in raw_cols, f"{name} is not a column of {raw.__name__}"
        raw_col = raw_cols[name]
        assert (col.dtype, col.nullable, col.description) == (
            raw_col.dtype,
            raw_col.nullable,
            raw_col.description,
        ), name


def test_clean_files_match_what_clean_data_saves() -> None:
    saved = [f.name for f in fields(Cleaned)] + [f.name for f in fields(ExtractedFeatures)]
    assert list(CLEAN_FILES) == saved


def test_slot_columns_cover_every_side_slot_and_feature() -> None:
    cols = slot_columns()
    assert len(cols) == 2 * N_SLOTS * len(PLAYER_SLOT_FEATURES)
    assert slot_column("h", 1, "xg90") == "h_player1_xg90"
    assert next(iter(cols)) == "h_player1_xg90"
    assert list(cols)[-1] == f"a_player{N_SLOTS}_{PLAYER_SLOT_FEATURES[-1]}"
    assert all(c.nullable and str(c.dtype) == "Float64" for c in cols.values())
    assert list(PLAYER_SLOTS.columns) == ["game_id", *cols]


def test_match_features_schema_adds_slots_to_the_model() -> None:
    model_cols = list(MatchFeaturesCols.to_schema().columns)
    assert list(match_features_schema().columns) == [*model_cols, *slot_columns()]


def test_dtype_returns_the_polars_dtype() -> None:
    assert dtype(MatchLeaguesCols, "away_lines") == pl.Int8
    assert dtype(EloCols, "match_date") == pl.Date


def test_read_clean_restores_dtypes_csv_loses(tmp_path: Path) -> None:
    # season "2014/15" would read back as a string anyway; match_date and the
    # goal columns would not keep their schema dtypes without the overrides.
    _elo(50.0).write_csv(tmp_path / "elo.csv")
    df = read_clean(tmp_path, "elo")
    assert df.schema["match_date"] == pl.Date
    assert df.schema["full_time_home_goals"] == pl.Int64


def test_read_clean_raises_on_invalid_file(tmp_path: Path) -> None:
    _elo(-50.0).write_csv(tmp_path / "elo.csv")
    with pytest.raises(ValueError, match="^elo:"):
        read_clean(tmp_path, "elo")


@pytest.fixture
def clean_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point check_clean at tmp_path, with only the elo file expected."""
    monkeypatch.setattr(schema.settings.clean, "out_dir", tmp_path)
    monkeypatch.setattr(schema, "CLEAN_FILES", {"elo": EloCols.to_schema()})
    return tmp_path


def test_check_clean_exits_0_when_every_file_is_valid(
    clean_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _elo(50.0).write_csv(clean_dir / "elo.csv")
    with pytest.raises(SystemExit) as info:
        schema.check_clean()
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith("ok      elo")


def test_check_clean_exits_1_on_invalid_file(
    clean_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _elo(-50.0).write_csv(clean_dir / "elo.csv")
    with pytest.raises(SystemExit) as info:
        schema.check_clean()
    assert info.value.code == 1
    assert capsys.readouterr().out.startswith("FAIL")


def test_check_clean_exits_1_on_missing_file(
    clean_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        schema.check_clean()
    assert info.value.code == 1
    assert capsys.readouterr().out.startswith("MISSING")
