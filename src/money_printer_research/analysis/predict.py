"""H/D/A probabilities for the fixtures not played yet, from the model `modeling` saved.

The fixtures are the unplayed rows (is_result False) of the Understat schedule, cleaned
like the played ones (clean_upcoming), from today on: an unplayed row dated in the past
is a cancelled game. Their features are built from match_features.csv,
the played matches, with nothing a fixture cannot know before kick-off:

- elo: each team's rating after its last played match (features.elo.fixture_elo).
- games_14d_all: the team's games in the 14 days before the fixture. Only league games
  are known here (played ones and earlier fixtures), so a cup or European game in that
  window is missed and the count can be low.
- player slots: the line-up is not known before kick-off, so each team gets the starters
  and the slot values of its last played match: the most likely XI, its form one game old.

Features the saved model does not use are not built; SavedModel.predict selects its own.
"""

from datetime import date, timedelta
from typing import cast

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.analysis.modeling import SavedModel, load_model
from money_printer_research.clean_data.match_stat import clean_upcoming
from money_printer_research.clean_data.read import read_match_stat
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.features.elo import fixture_elo, season_label, to_elo_input
from money_printer_research.schema import CleanedMatchCols, read_clean, slot_columns
from money_printer_research.utils import write_csv

GAMES_WINDOW = timedelta(days=14)
"""The window of games_14d_all: the 14 days before kick-off, kick-off day excluded."""

ID = ["game_id", "date", "league", "home_team", "away_team"]
"""The columns that name a fixture in the output."""

SIDES = (("h", "home_team"), ("a", "away_team"))
"""Each side's column prefix and team column."""


def _team_games(history: pl.DataFrame, fixtures: pl.DataFrame) -> pl.DataFrame:
    """One row per team per league game, played or fixture: team, day."""
    games = pl.concat([history.select(ID), fixtures.select(ID)])
    return pl.concat(
        games.select(pl.col(side).alias("team"), pl.col("date").dt.date().alias("day"))
        for side in ("home_team", "away_team")
    )


def games_in_window(history: pl.DataFrame, fixtures: pl.DataFrame) -> pl.DataFrame:
    """h_ and a_games_14d_all per fixture: each team's league games in GAMES_WINDOW before it."""
    games = _team_games(history, fixtures)
    out = fixtures.select(
        "game_id", pl.col("date").dt.date().alias("day"), "home_team", "away_team"
    )
    for prefix, side in SIDES:
        counts = (
            out.join(games, left_on=side, right_on="team")
            .filter(
                (pl.col("day_right") < pl.col("day"))
                & (pl.col("day_right") >= pl.col("day") - GAMES_WINDOW)
            )
            .group_by("game_id")
            .len(name=f"{prefix}_games_14d_all")
        )
        out = out.join(counts, on="game_id", how="left")
    return out.select("game_id", pl.col("^[ha]_games_14d_all$").fill_null(0).cast(pl.Float64))


def _slots(prefix: str) -> list[str]:
    """One side's starter slot columns: h_player1_xg90 ... for "h"."""
    return [c for c in slot_columns() if c.startswith(f"{prefix}_")]


def last_starters(history: pl.DataFrame) -> pl.DataFrame:
    """Per team: the starter slots of its last played match, without the side prefix."""
    sides = [
        history.select(
            pl.col(team).alias("team"),
            "date",
            *(pl.col(c).alias(c.removeprefix(f"{prefix}_")) for c in _slots(prefix)),
        )
        for prefix, team in SIDES
    ]
    return pl.concat(sides).sort("date").group_by("team").last().drop("date")


def fixture_features(history: pl.DataFrame, fixtures: pl.DataFrame) -> pl.DataFrame:
    """The fixtures with elo, games_14d_all and the player slots, built from `history`.

    `history` is match_features (played matches); `fixtures` is clean_upcoming's output.
    """
    played = to_elo_input(cast(DataFrame[CleanedMatchCols], history)).unwrap()
    elo = fixture_elo(
        played,
        fixtures.select(
            "game_id",
            season_label(pl.col("season_start")).alias("season"),
            pl.col("date").dt.date().alias("match_date"),
            "home_team",
            "away_team",
        ),
    ).select("game_id", "elo_home", "elo_away", "elo_diff")
    starters = last_starters(history)
    slots = [
        fixtures.select("game_id", pl.col(team).alias("team"))
        .join(starters, on="team", how="left")
        .drop("team")
        .rename(lambda c: c if c == "game_id" else f"{prefix}_{c}")
        for prefix, team in SIDES
    ]
    out = fixtures.select(*ID, "season_start")
    for part in [elo, games_in_window(history, fixtures), *slots]:
        out = out.join(part, on="game_id", how="left")
    return out.select(
        *ID,
        "season_start",
        "elo_home",
        "elo_away",
        "elo_diff",
        "h_games_14d_all",
        "a_games_14d_all",
        *slot_columns(),
    )


def predict_fixtures(
    model: SavedModel, history: pl.DataFrame, fixtures: pl.DataFrame
) -> pl.DataFrame:
    """One row per fixture, in kick-off order: who plays, then p_H, p_D, p_A."""
    frame = fixture_features(history, fixtures)
    proba = model.predict(frame)
    return (
        frame.select(ID)
        .join(proba, on="game_id")
        .select(*ID, "p_H", "p_D", "p_A")
        .sort("date", "game_id")
    )


def upcoming(settings: Settings, today: date | None = None) -> pl.DataFrame:
    """The fixtures still to come: unplayed in the Understat schedule, kick-off today or later.

    An unplayed fixture dated before today never will be played: Understat keeps games
    that were cancelled, such as the Ligue 1 2019/20 ones called off for COVID.
    """
    fixtures = read_match_stat(settings).and_then(clean_upcoming).expect("clean_upcoming")
    return still_to_come(fixtures, today or date.today())


def still_to_come(fixtures: pl.DataFrame, today: date) -> pl.DataFrame:
    """The fixtures whose kick-off day is today or later."""
    return fixtures.filter(pl.col("date").dt.date() >= today)


def main(settings: Settings = default_settings) -> pl.DataFrame:
    """Predict every unplayed fixture, write settings.modeling.predictions_file, return it."""
    fixtures = upcoming(settings)
    if fixtures.is_empty():
        print(
            "No fixtures still to come in the schedule. To predict the current season, set"
            " understat.end_year past it (end_year = 2027 in settings.toml, or"
            " UNDERSTAT__END_YEAR=2027, for 2026-27), then run"
            " collect_data and clean_data."
        )
        return pl.DataFrame()
    model = load_model(settings.modeling.out_file)
    history = read_clean(settings.clean.out_dir, "match_features")
    predictions = predict_fixtures(model, history, fixtures)
    out = settings.modeling.predictions_file
    write_csv(predictions, out)
    with pl.Config(float_precision=3, tbl_rows=50, tbl_width_chars=120):
        print(predictions)
    print(f"Saved {predictions.height} predictions to {out}")
    return predictions


def exec_predict() -> None:
    """Entry point `predict`: H/D/A probabilities for every unplayed fixture."""
    main()


__all__ = [
    "fixture_features",
    "games_in_window",
    "last_starters",
    "predict_fixtures",
    "still_to_come",
]
