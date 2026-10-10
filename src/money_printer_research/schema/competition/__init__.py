"""Competition schemas: Transfermarkt competitions and how they are grouped."""

from money_printer_research.schema.competition.comp_type import COMP_TYPES
from money_printer_research.schema.competition.raw import TransfermarktCompetitionsRaw, LEAGUE_INFO
from money_printer_research.schema.competition.selected import TransfermarktCompetitions

__all__ = [
    "COMP_TYPES",
    "LEAGUE_INFO",
    "TransfermarktCompetitions",
    "TransfermarktCompetitionsRaw",
]
