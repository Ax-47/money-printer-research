from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data.map_club_team import map_club_w_team
from money_printer_research.clean_data.match import join_match_w_stat
from money_printer_research.clean_data.match_leagues import curry_match_join_match_w_league
from money_printer_research.clean_data.match_stat import match_stat_monad
from money_printer_research.clean_data.read import (
    read_league_info,
    read_match_info,
    read_match_stat,
)

MAX_UNMATCHED_SHARE = 0.01
# from money_printer_research.clean_data.build_elo import add_elo
# from money_printer_research.clean_data.build_fatigue import player_fatigue, team_fatigue
# from money_printer_research.clean_data.build_player import build_player_feat, player_form
# from money_printer_research.clean_data.build_team_calendar import (
#     join_match_w_stat,
#     map_clubs,
#     standardize_games,
#     team_schedule_features,
# )
from money_printer_research.clean_data.rename_team_names import (
    find_missing_aliases,
    format_alias_lines,
    rename_team_names,
)
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.cool_stuff import Err, Ok, Result, all_ok, check, functor, monad
from money_printer_research.schema import CLEANED_MATCH, CLEANED_MATCH_STAT, CLUB_MAP, MATCH_LEAGUE
from money_printer_research.schema.selected import TransfermarktCompetitions as C
from money_printer_research.schema.selected import TransfermarktGames as G

# # from money_printer_research.schema.raw import PS, TM_COMPETITIONS, TM_GAMES, XG, validate_lazy
# from money_printer_research.schema.selected import UnderstatPlayer, UnderstatSchedule

type Frames = dict[str, pl.DataFrame]

NO_ALIAS_SOURCES = frozenset({"tm_games", "tm_competitions"})


# def _read_player(settings: Settings) -> pl.DataFrame:
#     # One sub-folder per league: <out_dir>/<league>/<season>.parquet
#     files = sorted(settings.understat.out_dir.glob("*/*.parquet"))
#     if not files:
#         raise FileNotFoundError(f"No parquet files in {settings.understat.out_dir}/<league>/")
#     return read_selected(files, PS)


def _alias_frames(frames: Frames) -> Frames:
    return {src: df for src, df in frames.items() if src not in NO_ALIAS_SOURCES}


def check_aliases[T: pl.DataFrame](df: T) -> Result[T]:
    """Err listing every team name in df that is missing from ALIASES; Ok(df) otherwise."""
    missing = find_missing_aliases(df)
    if missing.height:
        return Err(
            "check_aliases: not in ALIASES (target defaults to the name itself, "
            "edit it if the name is an alias):\n" + format_alias_lines({"xg": missing})
        )
    return Ok(df)


# @pa.check_types(lazy=True)
# def clean_player_stat(df: DataFrame[UnderstatPlayer]) -> DataFrame[PlayerStatCols]:
#     """Player match stats with season_start: "1415" -> 2014."""
#     out = df.with_columns(
#         (2000 + pl.col("season").str.slice(0, 2).cast(pl.Int32)).alias("season_start")
#     )
#     return cast(DataFrame[PlayerStatCols], out)
#
#
#
# def _season_label(start: pl.Expr) -> pl.Expr:
#     """2014 -> "2014/15"."""
#     return pl.concat_str(
#         start.cast(pl.String), pl.lit("/"), ((start + 1) % 100).cast(pl.String).str.zfill(2)
#     )
#
#
# def _add_elo(frames: Frames) -> Frames:
#     """Elo per league from 2014/15; treat the first season as warm-up."""
#     m = frames["matches"]
#     leagues = m["league"].unique().sort().to_list()
#     rated = pl.concat([add_elo(m.filter(pl.col("league") == lg)) for lg in leagues])
#     return {**frames, "matches": rated.sort("match_date", "game_id")}
#
#
# def _build_player_matches(frames: Frames) -> Frames:
#     players = frames["player_stat"]
#     games = frames["xg"].select(
#         "game_id", pl.col("date").dt.date().alias("match_date"), "home_team", "away_team"
#     )
#
#     # Many player rows per game, exactly one schedule row per game.
#     joined = players.join(games, on="game_id", how="inner", validate="m:1")
#
#     lost = players.height - joined.height
#     if lost:
#         raise ValueError(f"{lost} player rows have a game_id not in the schedule")
#
#     off_side = joined.filter(
#         (pl.col("team") != pl.col("home_team")) & (pl.col("team") != pl.col("away_team"))
#     )
#     if off_side.height:
#         names = sorted(off_side["team"].unique().to_list())
#         raise ValueError(f"team is neither home nor away in the schedule: {names}")
#
#     result = joined.with_columns(
#         (pl.col("team") == pl.col("home_team")).alias("is_home")
#     ).with_columns(
#         pl.when(pl.col("is_home"))
#         .then(pl.col("away_team"))
#         .otherwise(pl.col("home_team"))
#         .alias("opponent")
#     )
#     return {**frames, "player_matches": result}
#
#
# def _add_team_stats(frames: Frames) -> Frames:
#     """Post-match shots and cards per side, summed from Understat player rows."""
#     stats = ["shots", "yellow_cards", "red_cards"]
#     per_side = frames["player_matches"].group_by("game_id", "is_home").agg(pl.col(stats).sum())
#
#     def side(is_home: bool, prefix: str) -> pl.DataFrame:
#         return (
#             per_side.filter(pl.col("is_home") == is_home)
#             .drop("is_home")
#             .rename({c: f"{prefix}_{c}" for c in stats})
#         )
#
#     team = side(True, "home").join(side(False, "away"), on="game_id", how="full", coalesce=True)
#     matches = frames["matches"].join(team, on="game_id", how="left", validate="1:1")
#     return {**frames, "matches": matches}
#
#
# def _add_player_feat(frames: Frames) -> Frames:
#     form = player_form(frames["player_matches"])
#     fatigue = player_fatigue(form)
#     feat = build_player_feat(fatigue)
#     team = team_fatigue(fatigue)
#     matches = (
#         frames["matches"]
#         .join(feat, on="game_id", how="left", validate="1:1")
#         .join(team, on="game_id", how="left", validate="1:1")
#     )
#     return {**frames, "player_matches": fatigue, "matches": matches}
#
#
# def _add_fatigue(frames: Frames) -> Frames:
#     fatigue = player_fatigue(frames["player_matches"])
#     team = team_fatigue(fatigue)
#     matches = frames["matches"].join(team, on="game_id", how="left", validate="1:1")
#     return {**frames, "player_matches": fatigue, "matches": matches}
#
#
# def _add_team_calendar(frames: Frames) -> Frames:
#     """Schedule features from every club's full Transfermarkt calendar, per game_id."""
#     feat = team_schedule_features(frames["std_tm"], frames["clubs"], frames["stat_matches"])
#     matches = frames["matches"].join(feat, on="game_id", how="left", validate="1:1")
#     return {**frames, "matches": matches}
#
#
# def _validate_outputs(frames: Frames) -> Result[Frames]:
#     """Check every clean output against its schema, all problems at once, before saving."""
#     errors = []
#     for name, schema in CLEAN_FILES.items():
#         try:
#             validate_lazy(frames[name], schema, name)
#         except ValueError as e:
#             errors.append(str(e))
#     if errors:
#         return Err("\n\n".join(errors))
#     return Ok(frames)


#
# clean = compose(
#     lift(_read_all),
#     _check_aliases,
#     lift(functor(rename_team_names)),
#     lift(_clean_understat),
#     lift(_join_club_w_team),
#     lift(_join_transfermarkt_w_stat),
#     lift(_build_matches),
#     lift(_add_elo),
#     lift(_build_player_matches),
#     lift(_add_team_stats),
#     lift(_add_player_feat),
#     lift(_add_team_calendar),
#     _validate_outputs,
# )


def save_frames(frames: Frames, out_dir: Path) -> list[Path]:
    """Write each frame to <out_dir>/<key>.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, df in frames.items():
        path = out_dir / f"{key}.csv"
        df.write_csv(path)
        paths.append(path)
    return paths


@dataclass(frozen=True)
class Cleaned:
    """The clean outputs, one per CSV in settings.clean.out_dir (<field name>.csv)."""

    xg: DataFrame[CLEANED_MATCH_STAT]
    std_tm: DataFrame[MATCH_LEAGUE]
    clubs: DataFrame[CLUB_MAP]
    stat_matches: DataFrame[CLEANED_MATCH]

    def save(self, out_dir: Path) -> list[Path]:
        """Write each field to <out_dir>/<field name>.csv."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for field in fields(self):
            path = out_dir / f"{field.name}.csv"
            getattr(self, field.name).write_csv(path)
            paths.append(path)
        return paths


@monad
def clean(settings: Settings) -> Result[Cleaned]:
    """Run every step; Ok(Cleaned) or the first failing step's Err."""
    raw_match_stat = (Ok(settings) >> read_match_stat).if_err()
    cleaned_match_stat = match_stat_monad(raw_match_stat).if_err()
    xg = (
        Ok(cleaned_match_stat) >> check_aliases >> rename_team_names >> check(CLEANED_MATCH_STAT)
    ).if_err()  # 1

    selected_match, selected_league = all_ok(
        Ok(settings) >> read_match_info >> check(G), Ok(settings) >> read_league_info >> check(C)
    ).if_err()
    match_leagues = (
        Ok(selected_league)
        >> curry_match_join_match_w_league(selected_match)
        >> check(MATCH_LEAGUE)  # 2
    ).if_err()

    clubs = map_club_w_team(match_leagues, xg).if_err()
    cleaned_matches = join_match_w_stat(match_leagues, clubs, xg).if_err()

    unmatched = cleaned_matches[CLEANED_MATCH.tm_game_id].null_count()
    if unmatched > MAX_UNMATCHED_SHARE * cleaned_matches.height:  # 3
        return Err(
            f"join_match_w_stat: {unmatched:,} of {cleaned_matches.height:,} matches "
            "have no Transfermarkt game"
        )

    return Ok(Cleaned(xg=xg, std_tm=match_leagues, clubs=clubs, stat_matches=cleaned_matches))


def clean_data(settings: Settings = default_settings) -> Cleaned:
    """Entry point `clean_data`: run clean(), raise on Err, write every output as CSV."""
    cleaned = clean(settings).expect("clean_data")
    for path in cleaned.save(settings.clean.out_dir):
        print("saved", path, flush=True)
    return cleaned
