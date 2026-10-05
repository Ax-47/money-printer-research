import soccerdata as sd

from money_printer_research.config import Settings


def _collect_lineups(settings: Settings) -> None:
    cfg = settings.fbref
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
                fb = sd.FBref(leagues=league, seasons=[season], data_dir=cfg.cache_dir)
                df = fb.read_lineup().reset_index()
                df.to_parquet(out)
                print("saved", league, season, df.shape, flush=True)
            except Exception as e:
                print("FAILED", league, season, repr(e), flush=True)
