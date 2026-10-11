"""Backtest: would betting on the model's +EV picks against the bookmakers have paid?

Each season from settings.modeling.backtest_from is predicted walk-forward by the logistic
baseline (fitted on the seasons before it, as in modeling), then priced with the 1X2 odds
football-data.co.uk publishes per league and season, which collect_data downloads to
settings.modeling.odds_dir (data_collection.scrape_odds).

The odds are Pinnacle's closing prices: the sharpest widely quoted market, and what a bet
placed just before kick-off would get. Where they are missing the next available is used:
Pinnacle's opening price, the market average at closing, then Bet365's.

football-data names teams its own way ("Man United", "Paris SG"). team_names maps them to
the match_features names by which team each one plays alongside on the same days, so no
alias table is kept by hand.

For every game the pick is the outcome with the highest expected value of a 1-unit stake,
EV = p * odds - 1. The backtest table keeps every game with its pick, EV and profit, so a
strategy "bet the pick when its EV is above t" is a filter on it (summary).
"""

from pathlib import Path

import numpy as np
import polars as pl

from money_printer_research.analysis.feature_importance import (
    CLASSES,
    SELECTED,
    TARGET,
    columns,
    load,
)
from money_printer_research.analysis.modeling import logistic
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.utils import write_csv

OUTCOMES = ["H", "D", "A"]
ODDS_SOURCES = ["PSC{}", "PS{}", "AvgC{}", "B365{}"]
"""football-data odds columns per outcome, best first: Pinnacle closing, Pinnacle opening,
market average closing, Bet365 opening."""

GAME = ["league", "season_start", "home_team", "away_team"]
THRESHOLDS = (0.0, 0.02, 0.05, 0.1, 0.2)


def _read_one(path: Path, league: str, season_start: int) -> pl.DataFrame:
    df = pl.read_csv(
        path, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True
    )
    df = df.filter(pl.col("HomeTeam").is_not_null() & (pl.col("HomeTeam") != ""))
    date = pl.col("Date")
    odds = {
        f"odds_{k}": pl.coalesce(
            pl.col(c).cast(pl.Float64, strict=False)
            for src in ODDS_SOURCES
            if (c := src.format(k)) in df.columns
        )
        for k in OUTCOMES
    }
    return df.select(
        pl.lit(league).alias("league"),
        pl.lit(season_start).alias("season_start"),
        # Older files write two-digit years (16/08/15), newer ones four (16/08/2024).
        pl.when(date.str.len_chars() == 8)
        .then(date.str.to_date("%d/%m/%y", strict=False))
        .otherwise(date.str.to_date("%d/%m/%Y", strict=False))
        .alias("day"),
        pl.col("HomeTeam").alias("fd_home"),
        pl.col("AwayTeam").alias("fd_away"),
        **odds,
    )


def read_odds(odds_dir: Path) -> pl.DataFrame:
    """Every odds CSV collect_odds wrote: league, season_start, day, fd_home, fd_away, odds_H/D/A.

    The league is the file's folder, the season the start year of its name (2024-25.csv).
    """
    paths = sorted(odds_dir.glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"no odds in {odds_dir}; run `uv run collect_data` first")
    return pl.concat(_read_one(p, p.parent.name, int(p.stem[:4])) for p in paths)


def team_names(odds: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """league, fd_name, team: each football-data name and the games name it stands for.

    A name maps to the team it shares most game days with on the same side (home or
    away), in the same league and season. The other games of a match day add noise, but
    the right team shares every one of its days.
    """
    games = games.with_columns(day=pl.col("date").dt.date())
    pairs = pl.concat(
        odds.join(games, on=["league", "season_start", "day"]).select(
            "league", pl.col(f"fd_{side}").alias("fd_name"), pl.col(f"{side}_team").alias("team")
        )
        for side in ("home", "away")
    )
    return (
        pairs.group_by("league", "fd_name", "team")
        .len()
        .sort("len", "team", descending=[True, False])
        .group_by("league", "fd_name", maintain_order=True)
        .first()
        .drop("len")
    )


def attach_odds(games: pl.DataFrame, odds: pl.DataFrame) -> pl.DataFrame:
    """The games with odds_H/D/A, matched on league, season and both teams.

    Games with no match, or missing a price, are dropped.
    """
    names = team_names(odds, games)
    named = odds.join(
        names.rename({"fd_name": "fd_home", "team": "home_team"}), on=["league", "fd_home"]
    ).join(names.rename({"fd_name": "fd_away", "team": "away_team"}), on=["league", "fd_away"])
    odds_cols = [f"odds_{k}" for k in OUTCOMES]
    return games.join(named.select(*GAME, *odds_cols), on=GAME).drop_nulls(odds_cols)


def walk_forward(df: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """game_id with p_H, p_D, p_A for `seasons`, each from the logistic fitted on earlier ones."""
    cols, season = columns(SELECTED), pl.col("season_start")
    frames = []
    for s in seasons:
        test = df.filter(season == s)
        proba = logistic(df.filter(season < s), test, cols)
        frames.append(
            test.select("game_id").hstack(
                pl.DataFrame({f"p_{c}": proba[:, i] for i, c in enumerate(CLASSES)})
            )
        )
    return pl.concat(frames)


def picks(games: pl.DataFrame) -> pl.DataFrame:
    """The games (p_ and odds_ columns, and the result) with each one's pick, ev and profit.

    pick is the outcome with the highest EV; profit is what 1 unit on it returned.
    """
    p = games.select(f"p_{k}" for k in OUTCOMES).to_numpy()
    odds = games.select(f"odds_{k}" for k in OUTCOMES).to_numpy()
    ev = p * odds - 1
    best = ev.argmax(axis=1)
    rows = np.arange(len(best))
    pick = np.array(OUTCOMES)[best]
    won = pick == games[TARGET].to_numpy()
    return games.with_columns(
        pick=pl.Series(pick),
        ev=pl.Series(ev[rows, best]),
        profit=pl.Series(np.where(won, odds[rows, best] - 1, -1.0)),
    )


def log_loss_vs_market(table: pl.DataFrame) -> dict[str, float]:
    """Mean log loss of the model and of the bookmaker's margin-free probabilities."""
    actual = pl.col(TARGET)

    def loss(prob: dict[str, pl.Expr]) -> float:
        p = pl.coalesce(pl.when(actual == k).then(e) for k, e in prob.items())
        return float(table.select(-p.log().mean()).item())

    overround = pl.sum_horizontal(1 / pl.col(f"odds_{k}") for k in OUTCOMES)
    return {
        "model": loss({k: pl.col(f"p_{k}") for k in OUTCOMES}),
        "market": loss({k: 1 / pl.col(f"odds_{k}") / overround for k in OUTCOMES}),
    }


def summary(table: pl.DataFrame, min_ev: float, by: list[str] | None = None) -> pl.DataFrame:
    """Bets, hit rate, profit and ROI of betting 1 unit on every pick with ev > min_ev."""
    bets = table.filter(pl.col("ev") > min_ev)
    aggs = [
        pl.len().alias("bets"),
        (pl.col("profit") > 0).mean().alias("hit_rate"),
        pl.col("profit").sum().alias("profit"),
        pl.col("profit").mean().alias("roi"),
    ]
    return bets.group_by(by, maintain_order=True).agg(aggs).sort(by) if by else bets.select(aggs)


def cumulative_profit(table: pl.DataFrame, min_ev: float) -> pl.DataFrame:
    """day, profit: the running total of betting every pick with ev > min_ev, per match day."""
    return (
        table.filter(pl.col("ev") > min_ev)
        .group_by(pl.col("date").dt.date().alias("day"))
        .agg(pl.col("profit").sum())
        .sort("day")
        .with_columns(pl.col("profit").cum_sum())
    )


def backtest(settings: Settings = default_settings) -> pl.DataFrame:
    """The backtest table: every game from backtest_from with odds, probabilities and pick."""
    df = load(settings.clean.out_dir / "model_frame.csv")
    first = settings.modeling.backtest_from
    seasons = sorted(s for s in df["season_start"].unique() if s >= first)
    games = df.filter(pl.col("season_start") >= first).select("game_id", "date", *GAME, TARGET)
    games = attach_odds(games, read_odds(settings.modeling.odds_dir))
    return picks(games.join(walk_forward(df, seasons), on="game_id").sort("date", "game_id"))


def main(settings: Settings = default_settings) -> pl.DataFrame:
    """Run the backtest, print its summary, write settings.modeling.backtest_file."""
    table = backtest(settings)
    out = settings.modeling.backtest_file
    write_csv(table, out)
    losses = log_loss_vs_market(table)
    print(f"{table.height} games with odds, seasons {table['season_start'].min()} on")
    print(f"Log loss: model {losses['model']:.4f}, bookmaker {losses['market']:.4f}")
    with pl.Config(float_precision=3, tbl_rows=40):
        print(
            pl.concat(
                summary(table, t).select(pl.lit(t).alias("min_ev"), pl.all()) for t in THRESHOLDS
            )
        )
    print(f"Saved {out}")
    return table


def exec_backtest() -> None:
    """Entry point `backtest`: bet the model's +EV picks against historical odds."""
    main()


__all__ = [
    "attach_odds",
    "cumulative_profit",
    "log_loss_vs_market",
    "picks",
    "read_odds",
    "summary",
    "team_names",
    "walk_forward",
]
