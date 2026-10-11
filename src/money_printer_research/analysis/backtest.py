"""Backtest: would betting on the model's +EV picks against the bookmakers have paid?

Each season from settings.modeling.backtest_from is predicted walk-forward by the logistic
baseline (fitted on the seasons before it, as in modeling), then priced with the 1X2 odds
football-data.co.uk publishes per league and season, which collect_data downloads to
settings.modeling.odds_dir (data_collection.scrape_odds).

The odds are Pinnacle's closing prices: the sharpest widely quoted market, and what a bet
placed just before kick-off would get. Where they are missing the next available is used:
Pinnacle's early price, the market average at closing, then Bet365's early price
(ODDS_PREFERENCE; analysis.odds explains early and closing).

football-data names teams its own way ("Man United", "Paris SG"). team_names maps them to
the match_features names by which team each one plays alongside on the same days, so no
alias table is kept by hand.

For every game the pick is the outcome with the highest expected value of a 1-unit stake,
EV = p * odds - 1. The backtest table keeps every game with its pick, EV and profit, so a
strategy "bet the pick when its EV is above t" is a filter on it (summary).

A threshold picked by looking at the results flatters them, so two checks keep it honest:
walk_forward_thresholds picks each season's threshold from the seasons before it only,
and best_by_chance asks how often a model with no edge at all would show a best
(league, threshold) cell as good as the best one here.
"""

from collections.abc import Sequence
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
from money_printer_research.analysis.odds import one_price, read_prices
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.schema import BacktestCols, ThresholdSeasonCols
from money_printer_research.utils import write_csv

OUTCOMES = ["H", "D", "A"]
ODDS_PREFERENCE = [
    ("pinnacle", "close"),
    ("pinnacle", "early"),
    ("avg", "close"),
    ("bet365", "early"),
]
"""Where each game's 1X2 price comes from, best first (analysis.odds sources)."""

GAME = ["league", "season_start", "home_team", "away_team"]
THRESHOLDS = (0.0, 0.02, 0.05, 0.1, 0.2)
SLIDER_THRESHOLDS = tuple(i / 100 for i in range(31))
"""The thresholds the backtest page's slider offers, 0% to 30%."""
MIN_BETS = 30
"""Fewer bets than this swing too much for their z-score to mean anything."""


def read_odds(odds_dir: Path) -> pl.DataFrame:
    """league, season_start, day, fd_home, fd_away, odds_H/D/A: one 1X2 price per game.

    The price is the first source in ODDS_PREFERENCE that prices all three outcomes.
    """
    return one_price(read_prices(odds_dir), "1x2", ODDS_PREFERENCE)


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


def _odds_of_pick(table: pl.DataFrame) -> np.ndarray:
    """The odds of the outcome each game's pick backed."""
    odds = pl.coalesce(pl.when(pl.col("pick") == k).then(pl.col(f"odds_{k}")) for k in OUTCOMES)
    return table.select(odds).to_series().to_numpy()


def summary(table: pl.DataFrame, min_ev: float, by: list[str] | None = None) -> pl.DataFrame:
    """Bets, hit rate, profit and ROI of betting 1 unit on every pick with ev > min_ev."""
    bets = table.filter(pl.col("ev") > min_ev)
    aggs = [
        pl.len().alias("bets"),
        (pl.col("profit") > 0).mean().alias("hit_rate"),
        pl.col("profit").sum().alias("profit"),
        pl.col("profit").mean().alias("roi"),
        (pl.col("profit").std() / pl.len().sqrt()).alias("roi_se"),
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


def walk_forward_thresholds(
    table: pl.DataFrame, thresholds: Sequence[float] = SLIDER_THRESHOLDS
) -> pl.DataFrame:
    """Each season bet at a hindsight-free threshold (ThresholdSeasonCols).

    Before each season, pick the threshold that made the most on the seasons before it,
    as someone betting live would have to. The first season has nothing before it to
    choose from, so it is left out.
    """
    seasons = table["season_start"].to_numpy()
    ev = table["ev"].to_numpy()
    profit = table["profit"].to_numpy()

    rows = []
    for season in sorted(set(seasons.tolist()))[1:]:
        before = seasons < season
        chosen = max(thresholds, key=lambda t: profit[before & (ev > t)].sum())
        bets = profit[(seasons == season) & (ev > chosen)]
        roi = bets.mean() if bets.size else None
        rows.append((season, chosen, bets.size, bets.sum(), roi))
    table = pl.DataFrame(
        rows,
        schema={
            "season_start": pl.Int32,
            "min_ev": pl.Float64,
            "bets": pl.Int64,
            "profit": pl.Float64,
            "roi": pl.Float64,
        },
        orient="row",
    )
    return ThresholdSeasonCols.validate(table, lazy=True)


def _best_z_score(
    profit: np.ndarray, ev: np.ndarray, groups: list[np.ndarray], thresholds: Sequence[float]
) -> float:
    """The best ROI z-score over every (group, threshold) cell with at least MIN_BETS bets.

    The z-score is the ROI divided by its standard error: how many standard errors above
    breaking even the cell is.
    """
    best = -np.inf
    for in_group in groups:
        for threshold in thresholds:
            bets = profit[in_group & (ev > threshold)]
            if bets.size < MIN_BETS or (spread := bets.std(ddof=1)) == 0:
                continue
            best = max(best, bets.mean() / (spread / np.sqrt(bets.size)))
    return float(best)


def best_by_chance(
    table: pl.DataFrame,
    thresholds: Sequence[float] = SLIDER_THRESHOLDS,
    sims: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """How often the best (league, threshold) cell looks this good by luck alone.

    observed is the z-score of the best cell, where a cell is one league, or all of them,
    at one threshold. Each simulation replays the same picks at the same odds in a world
    where the model has no edge: every result is drawn from the bookmaker's margin-free
    probabilities. p_value is the share of simulations whose best cell scores at least
    as high as observed; far from 0, the best cell is no better than searching this many
    cells finds by chance.
    """
    leagues = table["league"].to_numpy()
    groups = [np.ones(table.height, bool), *(leagues == lg for lg in np.unique(leagues))]
    ev = table["ev"].to_numpy()
    pick_odds = _odds_of_pick(table)

    def best_z_score(won: np.ndarray) -> float:
        return _best_z_score(np.where(won, pick_odds - 1, -1.0), ev, groups, thresholds)

    observed = best_z_score((table["pick"] == table[TARGET]).to_numpy())

    implied = 1 / table.select(f"odds_{k}" for k in OUTCOMES).to_numpy()
    market_cdf = (implied / implied.sum(axis=1, keepdims=True)).cumsum(axis=1)
    pick = table["pick"].replace_strict({k: i for i, k in enumerate(OUTCOMES)}).to_numpy()
    rng = np.random.default_rng(seed)

    def no_edge_best_z_score() -> float:
        # A uniform draw lands past as many cumulative probabilities as the drawn
        # outcome's index: 0 is H, 1 is D, 2 is A.
        result = (rng.random(table.height)[:, None] > market_cdf).sum(axis=1)
        return best_z_score(result == pick)

    by_chance = np.array([no_edge_best_z_score() for _ in range(sims)])
    return {"observed": observed, "p_value": float(np.mean(by_chance >= observed))}


def backtest(settings: Settings = default_settings) -> pl.DataFrame:
    """The backtest table (BacktestCols): every game from backtest_from with odds and a pick."""
    df = load(settings.clean.out_dir / "model_frame.csv")
    first = settings.modeling.backtest_from
    seasons = sorted(s for s in df["season_start"].unique() if s >= first)
    games = df.filter(pl.col("season_start") >= first).select("game_id", "date", *GAME, TARGET)
    games = attach_odds(games, read_odds(settings.modeling.odds_dir))
    table = picks(games.join(walk_forward(df, seasons), on="game_id").sort("date", "game_id"))
    return BacktestCols.validate(table, lazy=True)


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
        print("Each season at the threshold that did best on the seasons before it:")
        print(walk_forward_thresholds(table))
    chance = best_by_chance(table)
    print(
        f"Best league and threshold: z-score {chance['observed']:.2f}. "
        f"A model with no edge scores at least as high {chance['p_value']:.0%} of the time."
    )
    print(f"Saved {out}")
    return table


def exec_backtest() -> None:
    """Entry point `backtest`: bet the model's +EV picks against historical odds."""
    main()


__all__ = [
    "attach_odds",
    "best_by_chance",
    "cumulative_profit",
    "log_loss_vs_market",
    "picks",
    "read_odds",
    "summary",
    "team_names",
    "walk_forward",
    "walk_forward_thresholds",
]
