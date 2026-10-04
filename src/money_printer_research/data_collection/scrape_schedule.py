import soccerdata as sd

from money_printer_research.config import Settings


def _collect_schedule(settings: Settings) -> None:
    cfg = settings.understat
    if cfg.schedule_file.exists():
        print("skip", cfg.schedule_file.name, flush=True)
        return

    cfg.schedule_file.parent.mkdir(parents=True, exist_ok=True)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(cfg.start_year, cfg.end_year)]
    us = sd.Understat(leagues=cfg.league, seasons=seasons, data_dir=cfg.cache_dir)
    sch = us.read_schedule().reset_index()
    sch.to_csv(cfg.schedule_file, index=False)
    print("saved", cfg.schedule_file.name, sch.shape, flush=True)
