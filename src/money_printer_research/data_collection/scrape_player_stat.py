import soccerdata as sd

from money_printer_research.config import Settings


def _collect_player_stats(settings: Settings) -> None:
    cfg = (settings or Settings()).understat
    # Kept separate from data/lineups so lineup files are not overwritten.
    out_dir = cfg.out_dir
    cache_dir = cfg.cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in range(cfg.start_year, cfg.end_year):
        season = f"{y}-{str(y + 1)[-2:]}"
        out = out_dir / f"{season}.parquet"
        if out.exists():
            print("skip", season, flush=True)
            continue
        try:
            us = sd.Understat(leagues=cfg.league, seasons=[season], data_dir=cache_dir)
            df = us.read_player_match_stats().reset_index()
            df.to_parquet(out)
            print("saved", season, df.shape, list(df.columns), flush=True)
        except Exception as e:
            print("FAILED", season, repr(e), flush=True)
