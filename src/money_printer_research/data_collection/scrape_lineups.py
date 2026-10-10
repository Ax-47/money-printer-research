import soccerdata as sd

from money_printer_research.config import Settings
from money_printer_research.data_collection.seasons import season_files


def collect_lineups(settings: Settings) -> None:
    """FBref line-ups, one parquet per league and season; existing files are skipped."""
    cfg = settings.fbref
    for league, season, out in season_files(
        cfg.out_dir, cfg.leagues, cfg.start_year, cfg.end_year
    ):
        if out.exists():
            print("skip", league, season, flush=True)
            continue
        try:
            fbref = sd.FBref(leagues=league, seasons=[season], data_dir=cfg.cache_dir)
            lineups = fbref.read_lineup().reset_index()
            lineups.to_parquet(out)
            print("saved", league, season, lineups.shape, flush=True)
        except Exception as e:
            print("FAILED", league, season, repr(e), flush=True)
