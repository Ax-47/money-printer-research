from pathlib import Path

import polars as pl

from money_printer_research.category_theory import Err, Ok, Result, compose, functor, lift
from money_printer_research.clean_data.build_elo import add_elo
from money_printer_research.clean_data.build_fatigue import player_fatigue, team_fatigue
from money_printer_research.clean_data.build_player import build_player_feat, player_form
from money_printer_research.clean_data.build_team_calendar import (
    map_clubs,
    game_info,
    map_games,
    standardize_games,
    team_schedule_features,
)
from money_printer_research.clean_data.rename_columns import snake_columns_pl
from money_printer_research.clean_data.rename_team_names import (
    find_missing_aliases,
    format_alias_lines,
    rename_team_names,
)
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.schema.raw import PS, TM_COMPETITIONS, TM_GAMES, XG, read_selected

type Frames = dict[str, pl.DataFrame]

NO_ALIAS_SOURCES = frozenset({"tm_games", "tm_competitions"})


def _read_xg(settings: Settings) -> pl.DataFrame:
    return read_selected(settings.understat.schedule_file, XG)


def _read_player(settings: Settings) -> pl.DataFrame:
    # One sub-folder per league: <out_dir>/<league>/<season>.parquet
    files = sorted(settings.understat.out_dir.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.understat.out_dir}/<league>/")
    return read_selected(files, PS)


def _read_all(settings: Settings) -> Frames:
    """Raw sources, only the columns the pipeline uses (schema.raw.SELECTED)."""
    tm_dir = settings.transfermarkt.out_dir
    return {
        "xg": _read_xg(settings),
        "player_stat": _read_player(settings),
        "tm_games": read_selected(tm_dir / "games.parquet", TM_GAMES),
        "tm_competitions": read_selected(tm_dir / "competitions.parquet", TM_COMPETITIONS),
    }


def _alias_frames(frames: Frames) -> Frames:
    return {src: df for src, df in frames.items() if src not in NO_ALIAS_SOURCES}


def _check_aliases(frames: Frames) -> Result[Frames]:
    per_source = functor(find_missing_aliases)(_alias_frames(frames))
    missing = {src: m for src, m in per_source.items() if m.height}
    if missing:
        return Err(
            "Not in ALIASES (target defaults to the name itself, "
            "edit it if the name is an alias):\n" + format_alias_lines(missing)
        )
    return Ok(frames)


_START_FROM_STR = 2000 + pl.col("season").str.slice(0, 2).cast(pl.Int32)

SEASON_START = {
    "player_stat": _START_FROM_STR,
    "xg": (2000 + pl.col("season") // 100).cast(pl.Int32),
}


def _add_season_start(frames: Frames) -> Frames:
    return {
        **frames,
        **{
            key: frames[key].with_columns(expr.alias("season_start"))
            for key, expr in SEASON_START.items()
        },
    }


def _season_label(start: pl.Expr) -> pl.Expr:
    """2014 -> "2014/15"."""
    return pl.concat_str(
        start.cast(pl.String), pl.lit("/"), ((start + 1) % 100).cast(pl.String).str.zfill(2)
    )


def _build_matches(frames: Frames) -> Frames:
    """One row per played match, every league, from the Understat schedule."""
    home, away = pl.col("home_goals"), pl.col("away_goals")
    base = (
        frames["xg"]
        .filter(pl.col("is_result"))
        .select(
            "league",
            _season_label(pl.col("season_start")).alias("season"),
            "season_start",
            "game_id",
            pl.col("date").dt.date().alias("match_date"),
            "home_team",
            "away_team",
            home.alias("full_time_home_goals"),
            away.alias("full_time_away_goals"),
            pl.when(home > away)
            .then(pl.lit("H"))
            .when(home < away)
            .then(pl.lit("A"))
            .otherwise(pl.lit("D"))
            .alias("full_time_result"),
            "home_xg",
            "away_xg",
        )
    )
    return {**frames, "matches": base}


def _add_elo(frames: Frames) -> Frames:
    """Elo per league from 2014/15; treat the first season as warm-up."""
    m = frames["matches"]
    leagues = m["league"].unique().sort().to_list()
    rated = pl.concat([add_elo(m.filter(pl.col("league") == lg)) for lg in leagues])
    return {**frames, "matches": rated.sort("match_date", "game_id")}


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


def _add_team_stats(frames: Frames) -> Frames:
    """Post-match shots and cards per side, summed from Understat player rows."""
    stats = ["shots", "yellow_cards", "red_cards"]
    per_side = frames["player_matches"].group_by("game_id", "is_home").agg(pl.col(stats).sum())

    def side(is_home: bool, prefix: str) -> pl.DataFrame:
        return (
            per_side.filter(pl.col("is_home") == is_home)
            .drop("is_home")
            .rename({c: f"{prefix}_{c}" for c in stats})
        )

    team = side(True, "home").join(side(False, "away"), on="game_id", how="full", coalesce=True)
    matches = frames["matches"].join(team, on="game_id", how="left", validate="1:1")
    return {**frames, "matches": matches}


def _add_player_feat(frames: Frames) -> Frames:
    form = player_form(frames["player_matches"])
    fatigue = player_fatigue(form)
    feat = build_player_feat(fatigue)
    team = team_fatigue(fatigue)
    matches = (
        frames["matches"]
        .join(feat, on="game_id", how="left", validate="1:1")
        .join(team, on="game_id", how="left", validate="1:1")
    )
    return {**frames, "player_matches": fatigue, "matches": matches}


def _add_fatigue(frames: Frames) -> Frames:
    fatigue = player_fatigue(frames["player_matches"])
    team = team_fatigue(fatigue)
    matches = frames["matches"].join(team, on="game_id", how="left", validate="1:1")
    return {**frames, "player_matches": fatigue, "matches": matches}


def _add_team_calendar(frames: Frames) -> Frames:
    tm = standardize_games(frames["tm_games"], frames["tm_competitions"])
    schedule = frames["xg"].filter(pl.col("is_result"))
    clubs = map_clubs(tm, schedule)
    bad = clubs.filter(
        pl.col("duplicate") | ((pl.col("coverage") < 0.9) & (pl.col("games") >= 10))
    )
    if bad.height:
        raise ValueError(f"suspicious club mappings:\n{bad}")
    game_map = map_games(tm, clubs, schedule)
    feat = team_schedule_features(tm, clubs, game_map)
    info = game_info(tm, game_map)
    # Understat keeps the score played on the pitch, Transfermarkt the official one,
    # so they differ where a result was overturned (forfeit, ineligible player).
    overturned = (pl.col("full_time_home_goals") != pl.col("tm_home_goals")) | (
        pl.col("full_time_away_goals") != pl.col("tm_away_goals")
    )
    matches = (
        frames["matches"]
        .join(feat, on="game_id", how="left", validate="1:1")
        .join(info, on="game_id", how="left", validate="1:1")
        .with_columns(overturned.fill_null(False).alias("result_overturned"))
        .drop("tm_home_goals", "tm_away_goals")
    )
    return {**frames, "matches": matches}


clean = compose(
    lift(_read_all),
    lift(functor(snake_columns_pl)),
    _check_aliases,
    # join
    lift(functor(rename_team_names)),
    lift(_add_season_start),
    lift(_build_matches),
    lift(_add_elo),
    lift(_build_player_matches),
    lift(_add_team_stats),
    lift(_add_player_feat),
    lift(_add_team_calendar),
)


def save_frames(frames: Frames, out_dir: Path) -> list[Path]:
    """Write each frame to <out_dir>/<key>.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, df in frames.items():
        path = out_dir / f"{key}.csv"
        df.write_csv(path)
        paths.append(path)
    return paths


def clean_data(settings: Settings = default_settings) -> Frames:
    match clean(settings):
        case Ok(frames):
            for path in save_frames(frames, settings.clean.out_dir):
                print("saved", path, flush=True)
            return frames
        case Err(reason, cause):
            raise ValueError(reason) from cause
