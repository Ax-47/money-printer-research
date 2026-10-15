from datetime import date
from pathlib import Path

import polars as pl

from money_printer_research.analysis.elo_sensitivity import (
    HOME_ADV_GRID,
    K_GRID,
    TUNE_SEASONS,
    _read_season,  # pyright: ignore[reportPrivateUsage]
    latex_grid,
)


def test_tuning_seasons_end_before_the_modelling_data() -> None:
    assert TUNE_SEASONS[0] == "2005/06"
    assert TUNE_SEASONS[-1] == "2014/15"


def test_read_season_parses_both_year_formats_and_drops_blank_rows(tmp_path: Path) -> None:
    csv = tmp_path / "2005-06.csv"
    csv.write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,13/08/05,Aston Villa,Bolton,2,2,D\n"
        "E0,14/08/2005,Everton,Man United,0,2,A\n"
        ",,,,,,\n"
    )
    df = _read_season(csv, 2005)
    assert df["season"].to_list() == ["2005/06", "2005/06"]
    assert df["match_date"].to_list() == [date(2005, 8, 13), date(2005, 8, 14)]
    assert df["full_time_result"].to_list() == ["D", "A"]


def test_latex_grid_bolds_the_default_setting() -> None:
    rows = [
        {"k": k, "home_adv": h, "season_regress": 0.0, "log_loss": 1.0 + k / 1000 + h / 1e5}
        for k in K_GRID
        for h in HOME_ADV_GRID
    ]
    tex = latex_grid(pl.DataFrame(rows))
    assert tex.count(r"\textbf") == 1
    assert r"\textbf{1.0154}" in tex  # k=15, home_adv=40
