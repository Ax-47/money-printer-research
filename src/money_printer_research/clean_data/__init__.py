from pathlib import Path

import polars as pl

from money_printer_research.category_theory import Err, Ok, Result, compose, functor, lift
from money_printer_research.clean_data.build_elo import add_elo
from money_printer_research.clean_data.build_fatigue import player_fatigue, team_fatigue
from money_printer_research.clean_data.build_player import build_player_feat, player_form
from money_printer_research.clean_data.rename_columns import snake_columns_pl
from money_printer_research.clean_data.rename_team_names import (
    find_missing_aliases,
    format_alias_lines,
    rename_team_names,
)
from money_printer_research.config import Settings, settings

type Frames = dict[str, pl.DataFrame]


def _read_kaggle(settings: Settings) -> pl.DataFrame:
    raw_kaggle_file = settings.kaggle.output_dir / "epl_final.csv"
    return pl.read_csv(raw_kaggle_file, try_parse_dates=True)


def _read_fbref(settings: Settings) -> pl.DataFrame:
    files = sorted(settings.fbref.out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.fbref.out_dir}")
    return pl.read_parquet(files)


def _read_xg(settings: Settings) -> pl.DataFrame:
    return pl.read_csv(settings.understat.schedule_file, try_parse_dates=True)


def _read_player(settings: Settings) -> pl.DataFrame:
    files = sorted(settings.understat.out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.fbref.out_dir}")
    return pl.read_parquet(files)


def _read_all(settings: Settings) -> Frames:
    return {
        "kaggle": _read_kaggle(settings),
        "fbref": _read_fbref(settings),
        "xg": _read_xg(settings),
        "player_stat": _read_player(settings),
    }


def _check_aliases(frames: Frames) -> Result[Frames]:
    per_source = functor(find_missing_aliases)(frames)
    missing = {src: m for src, m in per_source.items() if m.height}
    if missing:
        return Err(
            "Not in ALIASES (target defaults to the name itself, "
            "edit it if the name is an alias):\n" + format_alias_lines(missing)
        )
    return Ok(frames)


_START_FROM_STR = 2000 + pl.col("season").str.slice(0, 2).cast(pl.Int32)

SEASON_START = {
    "kaggle": pl.col("season").str.slice(0, 4).cast(pl.Int32),
    "fbref": _START_FROM_STR,
    "player_stat": _START_FROM_STR,
    "xg": (2000 + pl.col("season") // 100).cast(pl.Int32),
}


def _add_season_start(frames: Frames) -> Frames:
    return {
        key: df.with_columns(SEASON_START[key].alias("season_start")) for key, df in frames.items()
    }


def _build_matches(frames: Frames) -> Frames:
    keys = ["season_start", "home_team", "away_team"]
    xg = frames["xg"].select(*keys, "game_id", "home_xg", "away_xg")
    # validate="1:1" raises if any key repeats, instead of silently duplicating rows
    matches = frames["kaggle"].join(xg, on=keys, how="inner", validate="1:1")
    return {**frames, "matches": matches}


def _build_player_matches(frames: Frames) -> Frames:
    players = frames["player_stat"]
    games = frames["xg"].select(
        "game_id", pl.col("date").dt.date().alias("match_date"), "home_team", "away_team"
    )

    # Many player rows per game, exactly one schedule row per game.
    joined = players.join(games, on="game_id", how="inner", validate="m:1")

    lost = players.height - joined.height
    if lost:
        raise ValueError(f"{lost} player rows have a game_id not in the schedule")

    off_side = joined.filter(
        (pl.col("team") != pl.col("home_team")) & (pl.col("team") != pl.col("away_team"))
    )
    if off_side.height:
        names = sorted(off_side["team"].unique().to_list())
        raise ValueError(f"team is neither home nor away in the schedule: {names}")

    result = joined.with_columns(
        (pl.col("team") == pl.col("home_team")).alias("is_home")
    ).with_columns(
        pl.when(pl.col("is_home"))
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))
        .alias("opponent")
    )
    return {**frames, "player_matches": result}


def _add_player_feat(frames: Frames) -> Frames:
    form = player_form(frames["player_matches"])
    feat = build_player_feat(form)
    matches = frames["matches"].join(feat, on="game_id", how="left", validate="1:1")
    return {**frames, "player_matches": form, "matches": matches}


def _add_fatigue(frames: Frames) -> Frames:
    fatigue = player_fatigue(frames["player_matches"])
    team = team_fatigue(fatigue)
    matches = frames["matches"].join(team, on="game_id", how="left", validate="1:1")
    return {**frames, "player_matches": fatigue, "matches": matches}


def _add_elo(frames: Frames) -> Frames:
    keys = ["season_start", "home_team", "away_team"]
    elo = add_elo(frames["kaggle"]).select(*keys, "elo_home", "elo_away", "elo_diff")
    matches = frames["matches"].join(elo, on=keys, how="left", validate="1:1")
    return {**frames, "matches": matches}


clean = compose(
    lift(_read_all),
    lift(functor(snake_columns_pl)),
    _check_aliases,
    lift(functor(rename_team_names)),
    lift(_add_season_start),
    lift(_build_matches),
    lift(_add_elo),
    lift(_build_player_matches),
    lift(_add_player_feat),
    lift(_add_fatigue),
)


def save_frames(frames: Frames, out_dir: Path) -> list[Path]:
    """Write each frame to <out_dir>/<key>.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, df in frames.items():
        path = out_dir / f"{key}.csv"
        df.write_csv(path)
        paths.append(path)
    return paths


def clean_data() -> Frames:
    match clean(settings):
        case Ok(frames):
            for path in save_frames(frames, settings.clean.out_dir):
                print("saved", path, flush=True)
            return frames
        case Err(reason, cause):
            raise ValueError(reason) from cause
