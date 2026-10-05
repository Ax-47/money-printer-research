import soccerdata as sd

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings


def _collect_player_stats(settings: Settings = default_settings) -> None:
    """Understat player match stats, one parquet per league and season.

    Layout: <out_dir>/<league>/<season>.parquet. Existing files are skipped,
    so the run can be stopped and resumed.
    """
    cfg = settings.understat
    for league in cfg.leagues:
        league_dir = cfg.out_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        for y in range(cfg.start_year, cfg.end_year):
            season = f"{y}-{str(y + 1)[-2:]}"
            out = league_dir / f"{season}.parquet"
            if out.exists():
                print("skip", league, season, flush=True)
                continue
            try:
                us = sd.Understat(leagues=league, seasons=[season], data_dir=cfg.cache_dir)
                df = us.read_player_match_stats().reset_index()
                df.to_parquet(out)
                print("saved", league, season, df.shape, flush=True)
            except Exception as e:
                print("FAILED", league, season, repr(e), flush=True)
