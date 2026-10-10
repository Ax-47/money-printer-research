import polars as pl
import soccerdata as sd

from money_printer_research.config import Settings, UnderstatSettings
from money_printer_research.config import settings as default_settings
from money_printer_research.data_collection.checked import save_checked
from money_printer_research.data_collection.seasons import season_label, understat_season_code
from money_printer_research.schema.match.raw import UnderstatScheduleRaw


def _has_every_season(cfg: UnderstatSettings) -> bool:
    """True when the schedule file already holds every configured season."""
    if not cfg.schedule_file.exists():
        return False
    saved = set(pl.read_csv(cfg.schedule_file, columns=["season"])["season"])
    wanted = {understat_season_code(y) for y in range(cfg.start_year, cfg.end_year)}
    return wanted <= saved


def collect_schedule(settings: Settings = default_settings) -> None:
    """Understat schedule (goals and xG) for every league in settings, one CSV."""
    cfg = settings.understat
    if _has_every_season(cfg):
        print("skip", cfg.schedule_file.name, flush=True)
        return

    cfg.schedule_file.parent.mkdir(parents=True, exist_ok=True)
    seasons = [season_label(y) for y in range(cfg.start_year, cfg.end_year)]
    understat = sd.Understat(leagues=cfg.leagues, seasons=seasons, data_dir=cfg.cache_dir)
    schedule = understat.read_schedule().reset_index()
    save_checked(schedule, cfg.schedule_file, UnderstatScheduleRaw)
    print("saved", cfg.schedule_file.name, schedule.shape, flush=True)
