"""What staking does to a bankroll (issue #40).

The strategy suite scores profit per unit: every bet the same size. Real bettors size
bets, and how they size them changes how a run of results lands on the bankroll. This
replays a strategy's bets, in date order, under each staking plan, starting from a
bankroll of 1:

    flat            1% of the starting bankroll on every bet
    kelly           the Kelly fraction of the current bankroll, (p * odds - 1) / (odds - 1),
                    where p is the probability the strategy bets on (believed_p); half
    half_kelly      and quarter Kelly stake that much less. Kelly is the growth-optimal
    quarter_kelly   stake only if p is right; with an overconfident p it over-bets
    martingale      1% of the starting bankroll, doubled after every loss and reset after
                    a win, as long as the bankroll covers it

A bankroll that falls to 1% of where it started is ruined: it stops betting. Bets are
staked one after another; bets on the same day are not sized together, which flatters
Kelly slightly. Kelly stakes nothing where the strategy sees no edge (believed_p * odds
below 1), so a strategy that bets at Pinnacle's own prices is never staked by Kelly.

replay gives the one path history actually took. ruin_probabilities asks how likely each
outcome is: it resamples the strategy's bets with replacement into many alternative
histories of 1, 3 and 5 seasons and counts how often each plan loses half, or 90%, of
the bankroll at some point. Resampling single bets ignores any streakiness within a
season, so if anything it understates the risk.
"""

from typing import Any

import numpy as np
import polars as pl

from money_printer_research.analysis.strategies import run
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

PLANS = ["flat", "kelly", "half_kelly", "quarter_kelly", "martingale"]
FLAT_STAKE = 0.01
RUINED = 0.01
"""A bankroll at or below this share of the start has stopped betting."""

STAKED = [
    "favourite",
    "longshot",
    "always draw",
    "model EV, Pinnacle early [min_edge=0.02]",
    "model EV, best price early [min_edge=0.02]",
    "value 1X2, best price early [min_edge=0.02]",
]
"""The variants main reports on: the simple strategies and the most promising ones."""


def _kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """(p * odds - 1) / (odds - 1), or 0 where the bet has no edge under p."""
    p = np.nan_to_num(p, nan=0.0)
    return np.clip((p * odds - 1) / (odds - 1), 0, 1)


def simulate(returns: np.ndarray, odds: np.ndarray, believed: np.ndarray, plan: str) -> np.ndarray:
    """Bankroll after each bet, for every history at once.

    returns, odds, believed: (bets, histories) arrays; returns is the profit of a
    1-unit stake (odds - 1 for a win, -1 for a loss, in between for split AH stakes).
    Returns a (bets + 1, histories) array starting at 1.
    """
    n, histories = returns.shape
    bankroll = np.ones(histories)
    path = np.empty((n + 1, histories))
    path[0] = bankroll
    kelly_share = {"kelly": 1.0, "half_kelly": 0.5, "quarter_kelly": 0.25}.get(plan)
    stake_unit = np.full(histories, FLAT_STAKE)
    for i in range(n):
        alive = bankroll > RUINED
        if plan == "flat":
            stake = np.full(histories, FLAT_STAKE)
        elif kelly_share is not None:
            stake = kelly_share * _kelly_fraction(believed[i], odds[i]) * bankroll
        elif plan == "martingale":
            stake = stake_unit.copy()
        else:
            raise ValueError(f"unknown staking plan {plan!r}")
        stake = np.where(alive, np.minimum(stake, bankroll), 0.0)
        bankroll = bankroll + stake * returns[i]
        if plan == "martingale":
            stake_unit = np.where(returns[i] < 0, stake_unit * 2, FLAT_STAKE)
        path[i + 1] = bankroll
    return path


def _columns(bets: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """profit, odds and believed_p in date order, as (bets, 1) arrays."""
    ordered = bets.sort("day")

    def column(name: str) -> np.ndarray:
        return ordered[name].cast(pl.Float64).fill_null(np.nan).to_numpy()[:, None]

    return column("profit"), column("odds"), column("believed_p")


def _summary(path: np.ndarray) -> dict[str, float]:
    peak = np.maximum.accumulate(path, axis=0)
    return {
        "final": float(path[-1].mean()),
        "max_drawdown": float((1 - path / peak).max(axis=0).mean()),
        "lowest": float(path.min(axis=0).mean()),
    }


def replay(bets: pl.DataFrame) -> pl.DataFrame:
    """plan, final, max_drawdown, lowest: the history that happened, under each plan."""
    columns = _columns(bets)
    return pl.DataFrame([{"plan": plan, **_summary(simulate(*columns, plan))} for plan in PLANS])


def ruin_probabilities(
    bets: pl.DataFrame, horizons: tuple[int, ...] = (1, 3, 5), histories: int = 1000, seed: int = 0
) -> pl.DataFrame:
    """plan, seasons, p_lose_half, p_lose_90, median_final: over resampled histories.

    A history of h seasons is h times the strategy's mean bets per season, drawn with
    replacement from its bets. The p_ columns are shares of histories whose bankroll
    fell to 50% / 10% of the start at any point; their standard error is at most
    0.5 / sqrt(histories).
    """
    profit, odds, believed = (c[:, 0] for c in _columns(bets))
    per_season = len(profit) / bets["season_start"].n_unique()
    rng = np.random.default_rng(seed)
    rows = []
    for seasons in horizons:
        pick = rng.integers(0, len(profit), (round(per_season * seasons), histories))
        for plan in PLANS:
            path = simulate(profit[pick], odds[pick], believed[pick], plan)
            low = path.min(axis=0)
            rows.append(
                {
                    "plan": plan,
                    "seasons": seasons,
                    "p_lose_half": float((low <= 0.5).mean()),
                    "p_lose_90": float((low <= 0.1).mean()),
                    "median_final": float(np.median(path[-1])),
                }
            )
    return pl.DataFrame(rows)


def staking(bets: pl.DataFrame, variants: list[str] = STAKED) -> dict[str, pl.DataFrame]:
    """Run replay and ruin_probabilities for each variant, with a variant column."""
    replays, ruins = [], []
    for variant in variants:
        these = bets.filter(pl.col("variant") == variant)
        if these.is_empty():
            continue
        label = pl.lit(variant).alias("variant")
        replays.append(replay(these).select(label, pl.all()))
        ruins.append(ruin_probabilities(these).select(label, pl.all()))
    return {"replay": pl.concat(replays), "ruin": pl.concat(ruins)}


def main(settings: Settings = default_settings) -> dict[str, Any]:
    """Run the strategy suite, then stake its main variants every way and print the risk."""
    out = staking(run(settings)["bets"])
    with pl.Config(float_precision=3, tbl_rows=80, fmt_str_lengths=50, tbl_width_chars=200):
        print("The history that happened (bankroll starts at 1):")
        print(out["replay"])
        print("Over resampled histories:")
        print(out["ruin"])
    return out


def exec_staking() -> None:
    """Entry point `staking`: what each staking plan does to a bankroll."""
    main()


__all__ = ["PLANS", "replay", "ruin_probabilities", "simulate", "staking"]
