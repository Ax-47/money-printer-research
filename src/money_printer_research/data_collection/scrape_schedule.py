import polars as pl
import soccerdata as sd

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.data_collection.checked import save_checked
from money_printer_research.schema.raw import XG


def _collect_schedule(settings: Settings = default_settings) -> None:
    """Understat schedule (goals and xG) for every league in settings, one CSV."""
    cfg = settings.understat
    if cfg.schedule_file.exists():
        have = set(pl.read_csv(cfg.schedule_file, columns=["season"])["season"])
        want = {(y % 100) * 100 + (y + 1) % 100 for y in range(cfg.start_year, cfg.end_year)}
        if want <= have:
            print("skip", cfg.schedule_file.name, flush=True)
            return

    cfg.schedule_file.parent.mkdir(parents=True, exist_ok=True)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(cfg.start_year, cfg.end_year)]
    us = sd.Understat(leagues=cfg.leagues, seasons=seasons, data_dir=cfg.cache_dir)
    sch = us.read_schedule().reset_index()
    save_checked(sch, cfg.schedule_file, XG)
    print("saved", cfg.schedule_file.name, sch.shape, flush=True)
