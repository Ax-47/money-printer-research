from money_printer_research.config import settings
from money_printer_research.data_collection.scrape_lineups import (
    _collect_lineups as _collect_lineups,
)
from money_printer_research.data_collection.scrape_player_stat import _collect_player_stats
from money_printer_research.data_collection.scrape_schedule import _collect_schedule
from money_printer_research.data_collection.scrape_transfermarket import _collect_transfermarkt


def collect_data() -> None:
    """Fast and model-critical sources first; FBref lineups (slowest) last."""
    _collect_schedule(settings)  # Understat schedule, one request per league-season
    _collect_player_stats(settings)  # Understat, one page per match
    _collect_transfermarkt(settings)
