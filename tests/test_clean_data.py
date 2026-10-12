"""clean_data: the share of matches Transfermarkt cannot be joined to."""

import polars as pl

from money_printer_research.clean_data import unmatched_in_covered_seasons


def test_seasons_after_transfermarkt_do_not_count_as_unmatched() -> None:
    cleaned = pl.DataFrame(
        {"season_start": [2024, 2024, 2025, 2026, 2026], "tm_game_id": [1, None, 3, None, None]}
    )
    tm = pl.DataFrame({"season_start": [2024, 2025]})  # the dataset stops at 2025-26
    # 2024's missing game counts; 2026's two are past Transfermarkt's last season.
    assert unmatched_in_covered_seasons(cleaned, tm) == (1, 3, 2)
