"""The pre-registered strategy suite (issue #39): does any betting strategy win?

strategies.toml lists every strategy, written before any was run. Each one is a rule
for which bets to place; this module turns the football-data prices into the bets each
rule would have placed, settles them, and scores them.

Offers. For one market and one price (source, timing), offers has a row per game and
selection with:
    odds        the price the bet is placed at
    fair        Pinnacle's margin-free probability at the same moment: the sharpest
                estimate a bettor could see when placing the bet
    close_fair  Pinnacle's margin-free probability at kick-off, for the same line
    model_p     the market_and_form model's probability (1X2 only, from market.py)

Scores, per strategy variant and league (and all leagues together):
    roi         profit per unit staked, with a 95% interval
    close_ev    expected profit per unit if the closing price is right: close_fair *
                odds - 1. It measures edge with far less noise than profit, because it
                does not wait for the result; beating the closing price is the standard
                sign of a real edge.
    p_value     one-sided, for "ROI above zero"
    q_value     Benjamini-Hochberg over every cell in the suite: how many cells were
                tried is part of the answer.

walk_forward picks each strategy's parameter per season from the seasons before it,
as a bettor would have to.
"""

import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from money_printer_research.analysis.market import OUTCOMES, forecasts, margin_free, read_games
from money_printer_research.analysis.odds import GAME_KEY, read_prices, read_results
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.schema import MARKETS

MIN_BOOK = 0.97
"""Prices whose implied probabilities sum to less than this are dropped as data errors.

A sum below 1 is an arbitrage: backing every selection at these prices wins whatever
happens. Arbitrages of a percent or two do appear between bookmakers; one of 3% or more
is almost always a typo or a stale price nobody could take (football-data's best price
lists 22.0 on one side of an Asian handicap whose other side is 2.1). Keeping them would
credit a strategy with money no bettor could win.
"""

PARAMETERS = {
    "model_ev": "min_edge",
    "model_draw": "min_edge",
    "value": "min_edge",
    "odds_band": "band",
}
"""The kinds that take a parameter, and its name in strategies.toml."""


@dataclass(frozen=True)
class Strategy:
    """One entry of strategies.toml."""

    name: str
    kind: str
    market: str
    source: str
    timing: str
    values: list[Any] = field(default_factory=lambda: [None])
    """The parameter's values, one variant each; [None] for kinds without one."""
    selection: str | None = None


def load_suite(path: Path) -> tuple[int, list[Strategy]]:
    """test_from and the strategies in a strategies.toml."""
    config = tomllib.loads(path.read_text())
    suite = []
    for entry in config["strategy"]:
        parameter = PARAMETERS.get(entry["kind"])
        source, timing = entry["price"]
        suite.append(
            Strategy(
                name=entry["name"],
                kind=entry["kind"],
                market=entry["market"],
                source=source,
                timing=timing,
                values=list(entry[parameter]) if parameter else [None],
                selection=entry.get("selection"),
            )
        )
    return config["test_from"], suite


def file_commit(path: Path) -> str:
    """The last commit that changed path, so a report can name the suite it ran."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h %cs", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return out.stdout.strip() or "uncommitted"


# --- Offers: every price a strategy could take, with what is needed to judge it -----


def _wide(prices: pl.DataFrame, market: str, source: str, timing: str) -> pl.DataFrame:
    """One row per game (and line) with odds_<selection> from one source and timing."""
    selections = MARKETS[market]
    wide = (
        prices.filter(
            (pl.col("market") == market)
            & (pl.col("source") == source)
            & (pl.col("timing") == timing)
        )
        .with_columns(pl.col("line").fill_null(0.0))
        .pivot("selection", index=[*GAME_KEY, "line"], values="odds")
    )
    missing = [s for s in selections if s not in wide.columns]
    wide = wide.with_columns(pl.lit(None, pl.Float64).alias(s) for s in missing)
    wide = wide.drop_nulls(selections).rename({s: f"odds_{s}" for s in selections})
    book = pl.sum_horizontal(1 / pl.col(f"odds_{s}") for s in selections)
    return wide.filter(book >= MIN_BOOK)


def implausible_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """market, source, timing, games, dropped: how many games MIN_BOOK drops, per price."""
    rows = []
    for (market, source, timing), _ in prices.group_by("market", "source", "timing"):
        selections = MARKETS[str(market)]
        wide = (
            prices.filter(
                (pl.col("market") == market)
                & (pl.col("source") == source)
                & (pl.col("timing") == timing)
            )
            .pivot("selection", index=[*GAME_KEY, "line"], values="odds")
            .drop_nulls([s for s in selections])
        )
        if not all(s in wide.columns for s in selections):
            continue
        book = wide.select(pl.sum_horizontal(1 / pl.col(s) for s in selections)).to_series()
        rows.append((market, source, timing, wide.height, int((book < MIN_BOOK).sum())))
    return pl.DataFrame(
        rows, schema=["market", "source", "timing", "games", "dropped"], orient="row"
    ).sort("dropped", descending=True)


def _fair(prices: pl.DataFrame, market: str, timing: str, prefix: str) -> pl.DataFrame:
    """Pinnacle's margin-free probabilities at timing, as <prefix>_<selection>."""
    selections = MARKETS[market]
    pinnacle = _wide(prices, market, "pinnacle", timing)
    p = margin_free(pinnacle.select(f"odds_{s}" for s in selections).to_numpy(), "power")
    return pinnacle.select(*GAME_KEY, "line").with_columns(
        pl.Series(f"{prefix}_{s}", p[:, i]) for i, s in enumerate(selections)
    )


def offers(
    prices: pl.DataFrame,
    results: pl.DataFrame,
    model: pl.DataFrame | None,
    market: str,
    source: str,
    timing: str,
) -> pl.DataFrame:
    """Every (game, selection) priced by source at timing, long, with fair and close_fair.

    model, when given, has GAME_KEY and p_H/p_D/p_A; it is joined for 1X2 as model_p.
    """
    selections = MARKETS[market]
    games = (
        _wide(prices, market, source, timing)
        .join(_fair(prices, market, timing, "fair"), on=[*GAME_KEY, "line"])
        .join(_fair(prices, market, "close", "close"), on=[*GAME_KEY, "line"], how="left")
        .join(results, on=GAME_KEY)
    )
    if model is not None and market == "1x2":
        games = games.join(model, on=GAME_KEY)
    long = pl.concat(
        games.select(
            *GAME_KEY,
            "line",
            "home_goals",
            "away_goals",
            pl.lit(s).alias("selection"),
            pl.col(f"odds_{s}").alias("odds"),
            pl.col(f"fair_{s}").alias("fair"),
            pl.col(f"close_{s}").alias("close_fair"),
            (pl.col(f"p_{s}") if f"p_{s}" in games.columns else pl.lit(None, pl.Float64)).alias(
                "model_p"
            ),
        )
        for s in selections
    )
    return long.with_columns(profit=settle(market))


def _half(margin: pl.Expr) -> pl.Expr:
    """Profit of a unit stake on a handicap: win above 0, push at 0, lose below."""
    return pl.when(margin > 0).then(pl.col("odds") - 1).when(margin == 0).then(0.0).otherwise(-1.0)


def settle(market: str) -> pl.Expr:
    """Profit of 1 unit on the row's selection at its odds, given the score.

    Asian handicap: the line is the home side's goal start; a quarter line (-0.25, 0.75)
    splits the stake into two halves on the lines either side of it.
    """
    hg, ag, sel = pl.col("home_goals"), pl.col("away_goals"), pl.col("selection")
    if market == "1x2":
        result = (
            pl.when(hg > ag)
            .then(pl.lit("H"))
            .when(hg == ag)
            .then(pl.lit("D"))
            .otherwise(pl.lit("A"))
        )
        return pl.when(sel == result).then(pl.col("odds") - 1).otherwise(-1.0)
    if market == "ou25":
        over = (hg + ag) > 2.5
        won = pl.when(sel == "over").then(over).otherwise(~over)
        return pl.when(won).then(pl.col("odds") - 1).otherwise(-1.0)
    line = pl.col("line")
    margin = pl.when(sel == "home").then(hg - ag + line).otherwise(ag - hg - line)
    quarter = pl.when(((line * 4).round() % 2).abs() == 1).then(0.25).otherwise(0.0)
    return 0.5 * _half(margin - quarter) + 0.5 * _half(margin + quarter)


# --- The rules -------------------------------------------------------------------------


def bets(offer: pl.DataFrame, strategy: Strategy, value: Any) -> pl.DataFrame:
    """The offers the strategy takes with its parameter at value."""
    per_game = pl.col("odds")
    match strategy.kind:
        case "always":
            return offer.filter(pl.col("selection") == strategy.selection)
        case "favourite":
            return offer.filter(per_game == per_game.min().over(GAME_KEY))
        case "longshot":
            return offer.filter(per_game == per_game.max().over(GAME_KEY))
        case "odds_band":
            low, high = value
            return offer.filter((per_game >= low) & (per_game < high))
        case "value":
            return offer.filter(pl.col("fair") * per_game - 1 > value)
        case "model_draw":
            return offer.filter(
                (pl.col("selection") == "D") & (pl.col("model_p") - pl.col("fair") > value)
            )
        case "model_ev":
            ev = pl.col("model_p") * per_game - 1
            best = offer.drop_nulls("model_p").filter(ev == ev.max().over(GAME_KEY))
            return best.filter(pl.col("model_p") * pl.col("odds") - 1 > value)
    raise ValueError(f"unknown strategy kind {strategy.kind!r}")


def _variant(strategy: Strategy, value: Any) -> str:
    if value is None:
        return strategy.name
    shown = f"{value[0]:g}-{value[1]:g}" if isinstance(value, list) else f"{value:g}"
    return f"{strategy.name} [{PARAMETERS[strategy.kind]}={shown}]"


def all_bets(
    prices: pl.DataFrame,
    results: pl.DataFrame,
    model: pl.DataFrame,
    suite: list[Strategy],
    test_from: int,
) -> pl.DataFrame:
    """Every bet of every variant of every strategy, from test_from on.

    Columns: strategy, variant, value (the parameter, as text), league, season_start,
    day, selection, odds, profit, close_ev.
    """
    frames = []
    cache: dict[tuple[str, str, str], pl.DataFrame] = {}
    for strategy in suite:
        key = (strategy.market, strategy.source, strategy.timing)
        if key not in cache:
            cache[key] = offers(prices, results, model, *key).filter(
                pl.col("season_start") >= test_from
            )
        for value in strategy.values:
            frames.append(
                bets(cache[key], strategy, value).select(
                    pl.lit(strategy.name).alias("strategy"),
                    pl.lit(_variant(strategy, value)).alias("variant"),
                    pl.lit(str(value)).alias("value"),
                    "league",
                    "season_start",
                    "day",
                    "selection",
                    "odds",
                    "profit",
                    (pl.col("close_fair") * pl.col("odds") - 1).alias("close_ev"),
                )
            )
    return pl.concat(frames)


# --- Scores ----------------------------------------------------------------------------


def _with_all_leagues(table: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([table, table.with_columns(league=pl.lit("ALL"))])


def score(placed: pl.DataFrame) -> pl.DataFrame:
    """Per variant and league (and ALL): bets, profit, ROI and close_ev with intervals.

    p_value is one-sided for ROI > 0; q_value is Benjamini-Hochberg over every row.
    """
    se = pl.col("profit").std() / pl.len().sqrt()
    close_se = pl.col("close_ev").std() / pl.col("close_ev").count().sqrt()
    table = (
        _with_all_leagues(placed)
        .group_by("strategy", "variant", "league", maintain_order=True)
        .agg(
            pl.len().alias("bets"),
            pl.col("profit").sum().alias("profit"),
            pl.col("profit").mean().alias("roi"),
            (pl.col("profit").mean() - 1.96 * se).alias("roi_low"),
            (pl.col("profit").mean() + 1.96 * se).alias("roi_high"),
            pl.col("close_ev").mean().alias("close_ev"),
            (pl.col("close_ev").mean() - 1.96 * close_se).alias("close_ev_low"),
            (pl.col("close_ev").mean() + 1.96 * close_se).alias("close_ev_high"),
            (pl.col("profit").mean() / se).alias("z"),
        )
        .filter(pl.col("bets") >= 2)
    )
    p = norm.sf(table["z"].fill_nan(0).fill_null(0).to_numpy())
    return table.drop("z").with_columns(
        p_value=pl.Series(p), q_value=pl.Series(benjamini_hochberg(p))
    )


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """q-values: the smallest false discovery rate at which each test would be rejected."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(q)
    out[order] = q
    return out


def walk_forward(placed: pl.DataFrame) -> pl.DataFrame:
    """Per strategy and league (and ALL): each season at the value that did best before it.

    Columns: strategy, league, seasons, bets, profit, roi, roi_low, roi_high. The first
    scored season has nothing before it and is left out, for every strategy alike.
    """
    placed = _with_all_leagues(placed)
    by_season = placed.group_by("strategy", "league", "value", "season_start").agg(
        pl.col("profit").sum()
    )
    seasons = sorted(placed["season_start"].unique().to_list())
    chosen = []
    for season in seasons[1:]:
        best = (
            by_season.filter(pl.col("season_start") < season)
            .group_by("strategy", "league", "value")
            .agg(pl.col("profit").sum())
            .sort("profit", "value", descending=[True, False])
            .group_by("strategy", "league", maintain_order=True)
            .first()
        )
        chosen.append(
            best.select("strategy", "league", "value", pl.lit(season).alias("season_start"))
        )
    picks = pl.concat(chosen).with_columns(
        pl.col("season_start").cast(placed["season_start"].dtype)
    )
    taken = placed.join(picks, on=["strategy", "league", "value", "season_start"])
    se = pl.col("profit").std() / pl.len().sqrt()
    return (
        taken.group_by("strategy", "league", maintain_order=True)
        .agg(
            pl.col("season_start").n_unique().alias("seasons"),
            pl.len().alias("bets"),
            pl.col("profit").sum().alias("profit"),
            pl.col("profit").mean().alias("roi"),
            (pl.col("profit").mean() - 1.96 * se).alias("roi_low"),
            (pl.col("profit").mean() + 1.96 * se).alias("roi_high"),
        )
        .sort("strategy", "league")
    )


def model_probabilities(odds_dir: Path, test_from: int) -> pl.DataFrame:
    """GAME_KEY and p_H/p_D/p_A from market.py's market_and_form model, walk-forward."""
    forecast = forecasts(read_games(odds_dir), first_test_season=test_from)
    p = forecast["market_and_form"]
    return (
        forecast["games"]
        .select(GAME_KEY)
        .with_columns(pl.Series(f"p_{k}", p[:, i]) for i, k in enumerate(OUTCOMES))
    )


def run(settings: Settings = default_settings) -> dict[str, Any]:
    """Every strategy in settings.modeling.strategies_file: bets, scores, walk-forward."""
    path = settings.modeling.strategies_file
    test_from, suite = load_suite(path)
    odds_dir = settings.modeling.odds_dir
    placed = all_bets(
        read_prices(odds_dir),
        read_results(odds_dir),
        model_probabilities(odds_dir, test_from),
        suite,
        test_from,
    )
    return {
        "suite": path.name,
        "commit": file_commit(path),
        "test_from": test_from,
        "bets": placed,
        "scores": score(placed),
        "walk_forward": walk_forward(placed),
    }


def main(settings: Settings = default_settings) -> dict[str, Any]:
    """Run the suite and print every variant on all leagues, then the walk-forward ROI."""
    out = run(settings)
    scores = out["scores"]
    print(
        f"{out['suite']} at {out['commit']}: {scores.height} cells, seasons {out['test_from']} on"
    )
    shown = ["bets", "roi", "roi_low", "roi_high", "close_ev", "close_ev_low", "q_value"]
    with pl.Config(float_precision=3, tbl_rows=80, fmt_str_lengths=60, tbl_width_chars=200):
        print("All leagues, by ROI (close_ev: expected profit per unit at the closing price):")
        all_leagues = scores.filter(pl.col("league") == "ALL")
        print(all_leagues.sort("roi", descending=True).select("variant", *shown))
        print("Walk-forward (each season at the value that did best before it), all leagues:")
        print(out["walk_forward"].filter(pl.col("league") == "ALL").sort("roi", descending=True))
    return out


def exec_strategies() -> None:
    """Entry point `strategies`: run the pre-registered strategy suite."""
    main()


__all__ = [
    "Strategy",
    "all_bets",
    "benjamini_hochberg",
    "bets",
    "load_suite",
    "offers",
    "run",
    "score",
    "settle",
    "walk_forward",
]
