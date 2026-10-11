"""football-data.co.uk prices and results, as analysis.odds reads them.

    OddsCols    every price of every game, one row each
    ResultCols  every game's full-time score, to settle bets

A game is keyed by league, season_start, day and football-data's own team names
(fd_home, fd_away); analysis.backtest.team_names maps those to Understat names.
"""

import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F

MARKETS = {"1x2": ["H", "D", "A"], "ah": ["home", "away"], "ou25": ["over", "under"]}
"""Market -> its selections. ah is the Asian handicap; ou25 over/under 2.5 goals."""

TIMINGS = ["early", "close"]


class _GameKey(pa.DataFrameModel):
    league: F[str] = pa.Field(description='Understat league: "GER-Bundesliga".')
    season_start: F[pl.Int32] = pa.Field(description="Year the season starts: 2024 for 2024/25.")
    day: F[datetime.date] = pa.Field(description="Match date.")
    fd_home: F[str] = pa.Field(description='Home side as football-data names it: "Man United".')
    fd_away: F[str] = pa.Field(description="Away side as football-data names it.")


class OddsCols(_GameKey):
    """One price: one source's decimal odds on one selection, early or at the close."""

    market: F[str] = pa.Field(isin=list(MARKETS), description="1x2, ah or ou25.")
    selection: F[str] = pa.Field(
        isin=sorted({s for sel in MARKETS.values() for s in sel}),
        description="H/D/A for 1x2; home/away for ah; over/under for ou25.",
    )
    source: F[str] = pa.Field(
        description=(
            "Bookmaker (pinnacle, bet365, ...), betfair_exchange, or max / avg: the best "
            "and the mean price across the bookmakers football-data lists."
        )
    )
    timing: F[str] = pa.Field(
        isin=TIMINGS,
        description=(
            "close: at kick-off. early: football-data's pre-closing price, collected a day "
            "or two before (Friday afternoon for weekend games); not the true opening price."
        ),
    )
    line: F[float] = pa.Field(
        nullable=True,
        description=(
            "ah: the home side's goal start (-0.5 means home must win). ou25: 2.5. 1x2: null."
        ),
    )
    odds: F[float] = pa.Field(gt=1, description="Decimal odds: a winning 1-unit bet returns this.")


class ResultCols(_GameKey):
    """A played game's full-time score and shots on target, as football-data reports them."""

    home_goals: F[pl.Int32] = pa.Field(ge=0, description="Home goals at full time.")
    away_goals: F[pl.Int32] = pa.Field(ge=0, description="Away goals at full time.")
    home_shots_on_target: F[pl.Int32] = pa.Field(
        ge=0, nullable=True, description="Home shots on target; null where the file has none."
    )
    away_shots_on_target: F[pl.Int32] = pa.Field(
        ge=0, nullable=True, description="Away shots on target."
    )
