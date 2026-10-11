"""Match schemas: one row per match, from raw files to pre-match features."""

from money_printer_research.schema.match.raw import UnderstatScheduleRaw, TransfermarktGamesRaw
from money_printer_research.schema.match.selected import UnderstatSchedule, TransfermarktGames
from money_printer_research.schema.match.formation import (
    OUTFIELD,
    SHAPE_PATTERN,
    FormationCols,
    FORMATION_COLUMNS,
    GameFormationCols,
)
from money_printer_research.schema.match.clean import (
    ScheduleCols,
    MatchLeaguesCols,
    CleanedMatchCols,
)
from money_printer_research.schema.match.features import (
    MatchSideCols,
    OUTCOMES,
    EloInputCols,
    EloCols,
    PredictionCols,
    TeamFatigueCols,
    TeamScheduleCols,
    PLAYER_SLOT_FEATURES,
    N_SLOTS,
    slot_column,
    slot_columns,
    PLAYER_SLOTS,
    MatchFeaturesCols,
    MISSING_REASONS,
    MissingCols,
    match_features_schema,
)

__all__ = [
    "CleanedMatchCols",
    "EloCols",
    "EloInputCols",
    "FORMATION_COLUMNS",
    "FormationCols",
    "GameFormationCols",
    "MISSING_REASONS",
    "MatchFeaturesCols",
    "MatchLeaguesCols",
    "MatchSideCols",
    "MissingCols",
    "N_SLOTS",
    "OUTCOMES",
    "OUTFIELD",
    "PLAYER_SLOTS",
    "PLAYER_SLOT_FEATURES",
    "PredictionCols",
    "SHAPE_PATTERN",
    "ScheduleCols",
    "TeamFatigueCols",
    "TeamScheduleCols",
    "TransfermarktGames",
    "TransfermarktGamesRaw",
    "UnderstatSchedule",
    "UnderstatScheduleRaw",
    "match_features_schema",
    "slot_column",
    "slot_columns",
]
