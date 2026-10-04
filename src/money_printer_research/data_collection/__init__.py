from money_printer_research.config import settings
from money_printer_research.data_collection.get_matches import _collect_matches
from money_printer_research.data_collection.scrape_lineups import _collect_lineups
from money_printer_research.data_collection.scrape_player_stat import _collect_player_stats
from money_printer_research.data_collection.scrape_schedule import _collect_schedule


def collect_data():
    _collect_matches(settings)
    _collect_lineups(settings)
    _collect_player_stats(settings)
    _collect_schedule(settings)
