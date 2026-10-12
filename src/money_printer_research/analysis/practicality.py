"""What collecting an edge would take (issue #41).

The strategy suite finds one family with a positive expected value at the closing price:
line shopping, betting the best price across bookmakers when it beats Pinnacle's
margin-free probability. This asks what it would take a person to collect it.

Time. time_to_significance: how many bets, and seasons of the five leagues, before a
true edge of the observed size shows as profit two standard errors above zero. The edge
is taken both as the realised ROI and as close_ev (the expected profit at the closing
price, the better estimate). Football bets are noisy, about 1.8 units of standard
deviation per unit staked, so a few percent of edge takes tens of thousands of bets.

Limits. Bookmakers that offer the best price are mostly "soft" ones, and they restrict
or close accounts that win; Pinnacle and the exchanges do not, but their prices are not
the ones that beat the market. Kaunitz, Zhong and Kreiner (2017, "Beating the bookies
with their own numbers") ran this strategy with real money and had their accounts
limited within months. account_limits replays the bets with that rule: the bets are
spread over n accounts in turn, and an account is closed once it is up by a given number
of stakes. It reports what is collected before every account is closed.

Exchange. On the Betfair exchange no one closes winning accounts, but it charges
commission on net winnings. exchange_value runs the same rule at exchange prices after
commission (prices from 2024/25 only).
"""

from typing import Any

import numpy as np
import polars as pl

from money_printer_research.analysis.odds import read_prices, read_results
from money_printer_research.analysis.strategies import Strategy, bets, offers, run
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

EXAMINED = [
    "value 1X2, best price early [min_edge=0.02]",
    "value 1X2, best price close [min_edge=0.02]",
    "value AH, best price early [min_edge=0.02]",
    "value O/U 2.5, best price early [min_edge=0.02]",
    "model EV, best price early [min_edge=0.02]",
]
"""The variants with a positive close_ev whose interval is above zero."""

ACCOUNTS = (5, 10, 20)
LIMITS = (5.0, 20.0, 50.0)
"""Stakes of profit after which a soft bookmaker closes an account."""

COMMISSIONS = (0.02, 0.05)


def time_to_significance(placed: pl.DataFrame, variants: list[str] = EXAMINED) -> pl.DataFrame:
    """Per variant: bets per season, the edge, what it earns, and seasons until it shows.

    For the edge taken as the realised ROI and as close_ev: units_per_season is the
    expected profit of a season at 1 unit per bet, and seasons_needed the seasons until
    the mean is 2 standard errors above zero: with per-bet standard deviation sd and edge
    mu, that takes n >= (2 * sd / mu) ** 2 bets.
    """
    rows = []
    for variant in variants:
        these = placed.filter(pl.col("variant") == variant)
        if these.is_empty():
            continue
        profit = these["profit"].to_numpy()
        per_season = len(profit) / these["season_start"].n_unique()
        sd = profit.std(ddof=1)
        row: dict[str, Any] = {"variant": variant, "bets_per_season": round(per_season)}
        close_ev = these["close_ev"].drop_nulls().to_numpy()
        for name, edge in (("roi", float(profit.mean())), ("close_ev", float(close_ev.mean()))):
            needed = (2 * sd / edge) ** 2 if edge > 0 else float("inf")
            row |= {
                name: edge,
                f"units_per_season_{name}": edge * per_season,
                f"seasons_needed_{name}": needed / per_season,
            }
        rows.append(row)
    return pl.DataFrame(rows)


def account_limits(
    placed: pl.DataFrame,
    variant: str,
    accounts: tuple[int, ...] = ACCOUNTS,
    limits: tuple[float, ...] = LIMITS,
) -> pl.DataFrame:
    """accounts, limit, bets, profit, last_day: what is collected before all are closed.

    Bets go to the accounts in turn, skipping closed ones; an account closes once its
    own profit reaches limit stakes. With no limit, the profit is the whole history's.
    """
    ordered = placed.filter(pl.col("variant") == variant).sort("day")
    profit = ordered["profit"].to_numpy()
    days = ordered["day"].to_list()
    rows = [
        {
            "accounts": None,
            "limit": None,
            "bets": len(profit),
            "profit": float(profit.sum()),
            "last_day": days[-1] if days else None,
        }
    ]
    for n in accounts:
        for limit in limits:
            balance = np.zeros(n)
            open_accounts = list(range(n))
            turn, taken, total, last = 0, 0, 0.0, None
            for p, day in zip(profit, days, strict=True):
                if not open_accounts:
                    break
                account = open_accounts[turn % len(open_accounts)]
                balance[account] += p
                total += p
                taken += 1
                last = day
                if balance[account] >= limit:
                    open_accounts.remove(account)
                else:
                    turn += 1
            rows.append(
                {"accounts": n, "limit": limit, "bets": taken, "profit": total, "last_day": last}
            )
    return pl.DataFrame(rows)


def exchange_value(
    prices: pl.DataFrame,
    results: pl.DataFrame,
    min_edge: float = 0.02,
    commissions: tuple[float, ...] = COMMISSIONS,
) -> pl.DataFrame:
    """commission, bets, roi, close_ev: line shopping at Betfair exchange prices.

    A winning bet pays (odds - 1) * (1 - commission); the selection rule sees the price
    after commission too, as a bettor would.
    """
    rows = []
    for commission in commissions:
        offer = offers(prices, results, None, "1x2", "betfair_exchange", "early").with_columns(
            odds=1 + (pl.col("odds") - 1) * (1 - commission)
        )
        offer = offer.with_columns(
            profit=pl.when(pl.col("profit") > 0)
            .then(pl.col("odds") - 1)
            .otherwise(pl.col("profit"))
        )
        rule = Strategy("exchange", "value", "1x2", "betfair_exchange", "early")
        taken = bets(offer, rule, min_edge)
        rows.append(
            {
                "commission": commission,
                "bets": taken.height,
                "roi": taken["profit"].mean(),
                "close_ev": (taken["close_fair"] * taken["odds"] - 1).mean(),
            }
        )
    return pl.DataFrame(rows)


def main(settings: Settings = default_settings) -> dict[str, pl.DataFrame]:
    """Print time to significance, account limits and the exchange for the examined bets."""
    placed = run(settings)["bets"]
    odds_dir = settings.modeling.odds_dir
    out = {
        "time": time_to_significance(placed),
        "limits": account_limits(placed, EXAMINED[0]),
        "exchange": exchange_value(read_prices(odds_dir), read_results(odds_dir)),
    }
    with pl.Config(float_precision=3, tbl_rows=40, fmt_str_lengths=50, tbl_width_chars=200):
        print("Bets until the edge is 2 standard errors above zero:")
        print(out["time"])
        print(f"{EXAMINED[0]}, with soft bookmakers closing winning accounts:")
        print(out["limits"])
        print("The same rule on the Betfair exchange, after commission:")
        print(out["exchange"])
    return out


def exec_practicality() -> None:
    """Entry point `practicality`: what collecting the edge would take."""
    main()


__all__ = ["account_limits", "exchange_value", "time_to_significance"]
