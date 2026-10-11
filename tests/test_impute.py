import datetime

import polars as pl

from money_printer_research.features.calendar import MAX_GAP_DAYS
from money_printer_research.features.impute import PLAIN_VARIANT, model_frame

FORMATION = {
    "4-3-3 Attacking": ("4-3-3", "attacking", 4, 3, 3, 3),
    "4-2-3-1": ("4-2-3-1", None, 4, 5, 1, 4),
}
FIELDS = ("shape", "variant", "defenders", "midfielders", "forwards", "lines")


def _side(prefix: str, formations: list[str | None]) -> dict[str, pl.Series]:
    """The formation columns of one side, parsed the way encode_formation does."""
    parsed = [FORMATION[f] if f else (None,) * len(FIELDS) for f in formations]
    return {
        f"{prefix}_formation": pl.Series(formations, dtype=pl.String),
        **{
            f"{prefix}_{name}": pl.Series(
                [p[i] for p in parsed], dtype=pl.String if i < 2 else pl.Int8
            )
            for i, name in enumerate(FIELDS)
        },
    }


def _rows() -> pl.DataFrame:
    """Five games over two seasons; a and b meet three times, c and d twice.

    1 (2014): a-b, both formations known. 2 (2015): b-a, home b's formation missing.
    3 (2015): a-b, away b has no line-up in Understat. 4 (2015): c-d, c has no earlier
    formation, d's last game. 5 (2014): d-c, plain shapes.
    """
    return pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "season_start": pl.Series([2014, 2015, 2015, 2015, 2014], dtype=pl.Int32),
            "date": [datetime.datetime(2015, 5, d) for d in (1, 8, 15, 22, 2)],
            "home_team": ["a", "b", "a", "c", "d"],
            "away_team": ["b", "a", "b", "d", "c"],
            "h_min_7d": [450, 500, 470, 430, 420],
            "a_min_7d": [480, 460, None, 410, 400],
            "h_days_to_next_all": [7, 7, 7, None, 20],
            "a_days_to_next_all": [7, 7, 7, None, 20],
            **_side("home", ["4-3-3 Attacking", None, "4-2-3-1", None, "4-2-3-1"]),
            **_side("away", ["4-2-3-1", "4-3-3 Attacking", "4-2-3-1", "4-2-3-1", None]),
        }
    )


def _game(df: pl.DataFrame, game: int) -> dict[str, object]:
    return df.filter(pl.col("game_id") == game).row(0, named=True)


def test_model_frame_drops_no_lineup_and_warmup() -> None:
    out = model_frame(_rows())
    assert out["game_id"].to_list() == [2, 4]  # 3: no line-up; 1 and 5: 2014 warm-up


def test_formation_comes_from_the_teams_previous_match() -> None:
    game2 = _game(model_frame(_rows()), 2)
    # b played away in game 1 with 4-2-3-1; its plain shape then gets variant "none".
    assert game2["home_formation"] == "4-2-3-1"
    assert game2["home_variant"] == PLAIN_VARIANT
    assert game2["home_midfielders"] == 5
    assert game2["h_formation_imputed"] is True
    assert game2["away_formation"] == "4-3-3 Attacking"
    assert game2["a_formation_imputed"] is False


def test_formation_never_reads_a_later_match() -> None:
    out = model_frame(_rows(), warmup_seasons=0)
    # c's only other game (5) is earlier but has no formation for c either: stays null.
    game4 = _game(out, 4)
    assert game4["home_formation"] is None
    assert game4["h_formation_imputed"] is False
    # game 5's away side c: its next game (4) is later and has none, so null too.
    assert _game(out, 5)["away_formation"] is None


def test_last_game_gets_max_gap_days() -> None:
    game4 = _game(model_frame(_rows()), 4)
    assert game4["h_days_to_next_all"] == game4["a_days_to_next_all"] == MAX_GAP_DAYS


def test_model_frame_keeps_columns_and_adds_flags() -> None:
    rows = _rows()
    out = model_frame(rows)
    assert out.columns == [*rows.columns, "h_formation_imputed", "a_formation_imputed"]
    assert out.schema["home_defenders"] == pl.Int8
