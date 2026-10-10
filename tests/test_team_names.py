import polars as pl
import pytest

from money_printer_research.clean_data.rename_team_names import (
    check_aliases,
    find_missing_aliases,
    norm_team_names,
)


def test_norm_team_names_maps_aliases_to_one_name() -> None:
    df = pl.DataFrame(
        {"home_team": ["Man Utd", " Man. City "], "away_team": ["Manchester United", "man city"]}
    )
    out = norm_team_names(df)
    assert out["home_team"].to_list() == ["manchester united", "manchester city"]
    assert out["away_team"].to_list() == ["manchester united", "manchester city"]


def test_norm_team_names_raises_on_unknown_name() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        norm_team_names(pl.DataFrame({"team": ["Not A Real Club FC"]}))


def test_norm_team_names_needs_a_team_column() -> None:
    with pytest.raises(ValueError, match="no team column"):
        norm_team_names(pl.DataFrame({"club": ["man utd"]}))


def test_check_aliases_lists_missing_names() -> None:
    df = pl.DataFrame({"team": ["Man Utd", "Not A Real Club FC"]})
    assert find_missing_aliases(df)["key"].to_list() == ["not a real club fc"]
    result = check_aliases(df, source="xg")
    assert result.is_err()
    assert '"not a real club fc": "not a real club fc",  # xg' in str(result)


def test_check_aliases_ok_when_all_known() -> None:
    df = pl.DataFrame({"team": ["Man Utd"]})
    assert check_aliases(df).is_ok()
