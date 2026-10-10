import traceback

import pandas as pd
import soccerdata as sd

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.data_collection.checked import save_checked
from money_printer_research.data_collection.seasons import season_files
from money_printer_research.schema.player.raw import PLAYER_STAT


def _read_match_by_match(understat: sd.Understat, league: str, season: str) -> pd.DataFrame:
    """Read one match at a time, skipping matches Understat has no rosters for."""
    schedule = understat.read_schedule().reset_index()
    frames, skipped = [], []
    for game_id in schedule.loc[schedule["is_result"], "game_id"]:
        try:
            frames.append(understat.read_player_match_stats(match_id=int(game_id)))
        except AttributeError:
            skipped.append(int(game_id))
    print("skipped (no rosters)", league, season, skipped, flush=True)
    return pd.concat(frames)


def _read_season(understat: sd.Understat, league: str, season: str) -> pd.DataFrame:
    """Every player's match stats for the season, in one read when Understat allows it."""
    try:
        return understat.read_player_match_stats()
    except AttributeError:
        # One match with empty rosters breaks the whole season read.
        return _read_match_by_match(understat, league, season)


def collect_player_stats(settings: Settings = default_settings) -> None:
    """Understat player match stats, one parquet per league and season.

    Layout: <out_dir>/<league>/<season>.parquet. Existing files are skipped,
    so the run can be stopped and resumed.
    """
    cfg = settings.understat
    for league, season, out in season_files(
        cfg.out_dir, cfg.leagues, cfg.start_year, cfg.end_year
    ):
        if out.exists():
            print("skip", league, season, flush=True)
            continue
        try:
            understat = sd.Understat(leagues=league, seasons=[season], data_dir=cfg.cache_dir)
            stats = _read_season(understat, league, season).reset_index()
            save_checked(stats, out, PLAYER_STAT)
            print("saved", league, season, stats.shape, flush=True)
        except Exception as e:
            print("FAILED", league, season, repr(e), flush=True)
            traceback.print_exc()
