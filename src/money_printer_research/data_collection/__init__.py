from money_printer_research.config import settings
from money_printer_research.data_collection.scrape_lineups import collect_lineups
from money_printer_research.data_collection.scrape_player_stat import collect_player_stats
from money_printer_research.data_collection.scrape_schedule import collect_schedule
from money_printer_research.data_collection.scrape_transfermarket import collect_transfermarkt


def collect_data() -> None:
    """Collect every source the clean pipeline reads, fastest first.

    FBref line-ups are not part of this run (the slowest source, and nothing reads
    them yet); call collect_lineups(settings) to fetch them.
    """
    collect_schedule(settings)  # Understat schedule, one request per league-season
    collect_player_stats(settings)  # Understat, one page per match
    collect_transfermarkt(settings)


__all__ = [
    "collect_data",
    "collect_lineups",
    "collect_player_stats",
    "collect_schedule",
    "collect_transfermarkt",
]
