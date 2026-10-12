"""Every price football-data.co.uk publishes, as one long table (OddsCols).

A football-data file has one row per game and one column per price, named
<bookmaker><C if closing><selection>: PSCH is Pinnacle's closing price on a home win,
B365>2.5 Bet365's early price on over 2.5 goals, MaxCAHA the best closing price across
bookmakers on the away side of the Asian handicap. read_prices turns every such column
into rows of (market, selection, source, timing, line, odds), so a strategy asks for
"Pinnacle, closing, 1X2" instead of knowing column names.

Timing. Columns without the C are what football-data calls pre-closing: collected a day
or two before kick-off (Friday afternoon for weekend games). They are called "early"
here rather than "opening", because no bookmaker's true opening price is published.

Coverage changes over the years, and the table keeps what is there:
- Pinnacle 1X2, early and closing (PS*, PSC*): every season.
- Max and average across bookmakers: from 2019/20 as Max*/Avg*, early and closing;
  before that as BbMx*/BbAv* (Betbrain), early only.
- Asian handicap and over/under 2.5: early and closing from 2019/20, early only before.
- Smaller bookmakers come and go; COLUMN_SOURCES lists them.
"""

import re
from pathlib import Path

import polars as pl

from money_printer_research.schema import OddsCols, ResultCols

COLUMN_SOURCES = {
    "PS": "pinnacle",  # 1X2
    "P": "pinnacle",  # Asian handicap and over/under
    "B365": "bet365",
    "BW": "bet_and_win",
    "IW": "interwetten",
    "LB": "ladbrokes",
    "VC": "vc_bet",
    "WH": "william_hill",
    "SJ": "stan_james",
    "1XB": "1xbet",
    "BF": "betfair_sportsbook",
    "BFD": "betfred",
    "BMGM": "betmgm",
    "BV": "betvictor",
    "CL": "coral",
    "BFE": "betfair_exchange",
    "Max": "max",
    "Avg": "avg",
    "BbMx": "max",
    "BbAv": "avg",
}
"""football-data column prefix -> source. max and avg are across all bookmakers listed."""

_PREFIX = "|".join(sorted(map(re.escape, COLUMN_SOURCES), key=len, reverse=True))
_COLUMN = re.compile(
    rf"^(?P<prefix>{_PREFIX})(?P<closing>C?)"
    r"(?:(?P<x12>[HDA])|AH(?P<ah>[HA])|(?P<ou>[<>])2\.5)$"
)
"""<prefix><C?> then one of: H/D/A (1X2), AHH/AHA (Asian handicap), >2.5/<2.5."""

_AH_LINE = {"early": ("AHh", "BbAHh"), "close": ("AHCh",)}
"""Columns holding the Asian handicap line, the home side's goal start, by timing."""


def parse_column(name: str) -> dict[str, str] | None:
    """market, selection, source, timing of a price column; None if it is not a price.

    >>> parse_column("PSCH")
    {'market': '1x2', 'selection': 'H', 'source': 'pinnacle', 'timing': 'close'}
    >>> parse_column("B365>2.5")
    {'market': 'ou25', 'selection': 'over', 'source': 'bet365', 'timing': 'early'}
    """
    m = _COLUMN.match(name)
    if m is None:
        return None
    if m["x12"]:
        market, selection = "1x2", m["x12"]
    elif m["ah"]:
        market, selection = "ah", {"H": "home", "A": "away"}[m["ah"]]
    else:
        market, selection = "ou25", {">": "over", "<": "under"}[m["ou"]]
    # "PS" prices only 1X2 and "P" only the other markets: PSH is Pinnacle's home win.
    if (m["prefix"], market == "1x2") in {("PS", False), ("P", True)}:
        return None
    return {
        "market": market,
        "selection": selection,
        "source": COLUMN_SOURCES[m["prefix"]],
        "timing": "close" if m["closing"] else "early",
    }


def _read_file(path: Path) -> pl.DataFrame:
    """One football-data CSV as strings, without its trailing blank rows."""
    df = pl.read_csv(
        path, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True
    )
    df = df.rename(lambda c: c.lstrip("﻿"))
    return df.filter(pl.col("HomeTeam").is_not_null() & (pl.col("HomeTeam") != ""))


def _game_keys(league: str, season_start: int) -> list[pl.Expr]:
    date = pl.col("Date")
    return [
        pl.lit(league).alias("league"),
        pl.lit(season_start, pl.Int32).alias("season_start"),
        # Older files write two-digit years (16/08/15), newer ones four (16/08/2024).
        pl.when(date.str.len_chars() == 8)
        .then(date.str.to_date("%d/%m/%y", strict=False))
        .otherwise(date.str.to_date("%d/%m/%Y", strict=False))
        .alias("day"),
        pl.col("HomeTeam").alias("fd_home"),
        pl.col("AwayTeam").alias("fd_away"),
    ]


GAME_KEY = ["league", "season_start", "day", "fd_home", "fd_away"]


def _prices_of_file(path: Path, league: str, season_start: int) -> pl.DataFrame:
    df = _read_file(path)
    frames = []
    for column in df.columns:
        price = parse_column(column)
        if price is None:
            continue
        line: pl.Expr = pl.lit(None, pl.Float64)
        if price["market"] == "ou25":
            line = pl.lit(2.5)
        elif price["market"] == "ah":
            line_columns = [c for c in _AH_LINE[price["timing"]] if c in df.columns]
            if not line_columns:
                continue
            line = pl.col(line_columns[0]).cast(pl.Float64, strict=False)
        frames.append(
            df.select(
                *_game_keys(league, season_start),
                *(pl.lit(v).alias(k) for k, v in price.items()),
                line.alias("line"),
                pl.col(column).cast(pl.Float64, strict=False).alias("odds"),
            )
        )
    return pl.concat(frames).filter(pl.col("odds") > 1)


def _files(odds_dir: Path) -> list[tuple[Path, str, int]]:
    """Every file collect_odds wrote: path, league (its folder), season (2024-25 -> 2024)."""
    paths = sorted(odds_dir.glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"no odds in {odds_dir}; run `uv run collect_data` first")
    return [(p, p.parent.name, int(p.stem[:4])) for p in paths]


def read_prices(odds_dir: Path) -> pl.DataFrame:
    """Every price in every file under odds_dir, one row each (OddsCols).

    A source that lists the same selection twice for a game (rare duplicate rows in the
    files) keeps its first price.
    """
    prices = pl.concat(_prices_of_file(*f) for f in _files(odds_dir))
    key = [*GAME_KEY, "market", "selection", "source", "timing"]
    return OddsCols.validate(prices.unique(key, keep="first", maintain_order=True), lazy=True)


RESULT_COLUMNS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
}
"""football-data column -> ResultCols column."""


def _results_of_file(path: Path, league: str, season_start: int) -> pl.DataFrame:
    df = _read_file(path)
    return df.select(
        *_game_keys(league, season_start),
        *(
            (pl.col(column) if column in df.columns else pl.lit(None))
            .cast(pl.Int32, strict=False)
            .alias(name)
            for column, name in RESULT_COLUMNS.items()
        ),
    )


def read_results(odds_dir: Path) -> pl.DataFrame:
    """Full-time score and shots on target of every game in the odds files (ResultCols)."""
    results = (
        pl.concat(_results_of_file(*f) for f in _files(odds_dir))
        .drop_nulls(["home_goals", "away_goals"])
        .unique(GAME_KEY, keep="first", maintain_order=True)
    )
    return ResultCols.validate(results, lazy=True)


def one_price(
    prices: pl.DataFrame, market: str, preference: list[tuple[str, str]]
) -> pl.DataFrame:
    """One price per game and selection: the first of (source, timing) in preference.

    The result is wide, one row per game, with an odds_<selection> column for each
    selection, and a game only appears if every selection has a price from the same
    (source, timing). Asian handicap lines differ by source, so this is for 1X2 and O/U.
    """
    ranks = pl.DataFrame(
        [(source, timing, i) for i, (source, timing) in enumerate(preference)],
        schema=["source", "timing", "rank"],
        orient="row",
    )
    ranked = prices.filter(pl.col("market") == market).join(ranks, on=["source", "timing"])
    selections = ranked["selection"].unique().len()
    complete = (
        ranked.group_by(*GAME_KEY, "rank")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") == selections)
        .group_by(GAME_KEY)
        .agg(pl.col("rank").min())
    )
    return (
        ranked.join(complete, on=[*GAME_KEY, "rank"])
        .pivot("selection", index=GAME_KEY, values="odds")
        .rename(lambda c: c if c in GAME_KEY else f"odds_{c}")
        .sort(GAME_KEY)
    )
