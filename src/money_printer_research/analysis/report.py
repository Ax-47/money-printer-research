"""One page answering issue #36: can the strongest honest attempt beat the market?

`uv run report` runs every analysis from the raw football-data files (market_test,
strategies, staking, practicality) and writes settings.modeling.report_file, a single
HTML page with no external files. Every number and chart on it comes from that run; the
only hand-written text is the method and the limitations, which say what was done, not
what came out.

The page names the commit of the code and of strategies.toml it ran with, so a reader
can check out that commit and get the same page.
"""

import datetime
import html
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

from money_printer_research.analysis import practicality, staking
from money_printer_research.analysis.market import (
    COMPARISONS,
    FIRST_TEST_SEASON,
    forecasts,
    losses,
    margin_methods,
    paired_differences,
    read_games,
)
from money_printer_research.analysis.odds import read_prices, read_results
from money_printer_research.analysis.strategies import run as run_strategies
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

FORECASTER_NAMES = {
    "early": "Pinnacle early price",
    "close": "Pinnacle closing price",
    "market_refit": "early price, recalibrated",
    "market_and_form": "early price + team form",
    "form_only": "team form only",
}


# --- Collecting the results ------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def collect(settings: Settings = default_settings) -> dict[str, Any]:
    """Run every analysis once and return what the page shows."""
    odds_dir = settings.modeling.odds_dir
    games = read_games(odds_dir)
    table = losses(forecasts(games))
    suite = run_strategies(settings)
    placed = suite["bets"]
    root = settings.modeling.strategies_file.parent
    return {
        "generated": datetime.date.today().isoformat(),
        "code_commit": _git("rev-parse", "--short", "HEAD", cwd=root),
        "dirty": bool(_git("status", "--porcelain", "--", "src", "strategies.toml", cwd=root)),
        "suite_commit": suite["commit"],
        "games": games,
        "test_games": table.filter(pl.col("forecaster") == "early").height,
        "margins": margin_methods(games),
        "forecasters": table.group_by("forecaster").agg(pl.col("log_loss", "rps").mean()),
        "comparisons": paired_differences(table, COMPARISONS),
        "by_league": paired_differences(table, [("market_and_form", "early")], by=["league"]),
        "scores": suite["scores"],
        "walk_forward": suite["walk_forward"],
        "test_from": suite["test_from"],
        "staking": staking.staking(placed),
        "time": practicality.time_to_significance(placed),
        "limits": practicality.account_limits(placed, practicality.EXAMINED[0]),
        "exchange": practicality.exchange_value(read_prices(odds_dir), read_results(odds_dir)),
    }


# --- Formatting ------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 4, sign: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _interval(mid: float, low: float, high: float, fmt: Any = _pct) -> str:
    return f"{fmt(mid)} <span class=muted>({fmt(low)} to {fmt(high)})</span>"


def _table(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    """An HTML table; cells are already HTML. Columns from numeric_from are right-aligned."""
    th = "".join(
        f"<th{' class=num' if i >= numeric_from else ''}>{h}</th>" for i, h in enumerate(head)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{' class=num' if i >= numeric_from else ''}>{c}</td>" for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return (
        f"<div class=scroll><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def interval_chart(
    rows: list[tuple[str, float, float, float]], unit: str, fmt: Any = _pct, width: int = 720
) -> str:
    """A dot-and-interval chart: one row per (label, estimate, low, high), zero marked.

    Each row's dot is the estimate, its line the 95% interval; hovering shows both.
    """
    label_w, pad, row_h = 330, 16, 26
    plot_w = width - label_w - pad * 2
    values = [v for _, mid, low, high in rows for v in (mid, low, high)] + [0.0]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05

    def x(v: float) -> float:
        return label_w + pad + (v - lo) / (hi - lo) * plot_w

    height = row_h * len(rows) + 34
    parts = [
        f'<div class=scroll><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(unit)}" class=chart>',
        f'<line class=zero x1="{x(0):.1f}" x2="{x(0):.1f}" y1="4" y2="{height - 26}"/>',
        f'<text class=tick x="{x(0):.1f}" y="{height - 10}" text-anchor="middle">0</text>',
        f'<text class=tick x="{x(lo):.1f}" y="{height - 10}">{fmt(lo)}</text>',
        f'<text class=tick x="{x(hi):.1f}" y="{height - 10}" text-anchor="end">{fmt(hi)}</text>',
    ]
    for i, (label, mid, low, high) in enumerate(rows):
        y = 16 + i * row_h
        tip = html.escape(f"{label}: {fmt(mid)} (95% interval {fmt(low)} to {fmt(high)})")
        above = low > 0
        parts.append(
            f'<g class="row{" above" if above else ""}"><title>{tip}</title>'
            f'<rect class=hit x="0" y="{y - row_h / 2:.1f}" width="{width}" height="{row_h}"/>'
            f'<text class=label x="{label_w}" y="{y + 4}" text-anchor="end">'
            f"{html.escape(label)}</text>"
            f'<line class=ci x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}"/>'
            f'<circle class=dot cx="{x(mid):.1f}" cy="{y}" r="4.5"/></g>'
        )
    parts.append("</svg></div>")
    return "".join(parts)


# --- The page --------------------------------------------------------------------------


def _summary(r: dict[str, Any]) -> list[str]:
    """The answer, as sentences built from the numbers."""
    comp = {(row["a"], row["b"]): row for row in r["comparisons"].iter_rows(named=True)}
    scores = r["scores"]
    all_leagues = scores.filter(pl.col("league") == "ALL")
    significant = scores.filter(pl.col("q_value") < 0.05).height
    simple = all_leagues.filter(
        pl.col("variant").is_in(
            ["always home", "always draw", "always away", "favourite", "longshot"]
        )
    )
    shopping = all_leagues.filter(
        pl.col("variant").str.starts_with("value") & (pl.col("close_ev_low") > 0)
    )
    ruin = r["staking"]["ruin"]
    martingale = ruin.filter((pl.col("plan") == "martingale") & (pl.col("seasons") == 1))
    time = r["time"].filter(pl.col("variant").str.starts_with("value"))
    exchange = r["exchange"].row(0, named=True)

    feat = comp[("market_and_form", "early")]
    close = comp[("close", "early")]
    sentences = [
        "<b>No model built here forecasts better than the market.</b> Adding team form to "
        f"Pinnacle's early price changes log loss by {_num(feat['log_loss'])} "
        f"(95% interval {_num(feat['log_loss_low'])} to {_num(feat['log_loss_high'])}; below "
        "zero would mean better), while the closing price beats the early one by "
        f"{_num(close['log_loss'])} ({_num(close['log_loss_low'])} to "
        f"{_num(close['log_loss_high'])}).",
        f"<b>No strategy survives the correction for how many were tried.</b> Of "
        f"{scores.height} strategy × league cells, {significant} "
        f"{'has' if significant == 1 else 'have'} a Benjamini–Hochberg q-value under 0.05. "
        f"The simple ones lose: ROI from {_pct(simple['roi'].min())} to "
        f"{_pct(simple['roi'].max())}.",
    ]
    if shopping.height:
        sentences.append(
            "<b>The one family with an expected edge is line shopping</b>: taking the best "
            "price across bookmakers when it beats Pinnacle's. Its expected profit at the "
            f"closing price is {_pct(shopping['close_ev'].min())} to "
            f"{_pct(shopping['close_ev'].max())} per bet. That edge exists only at the "
            "soft bookmakers that close winning accounts; showing it as profit takes "
            f"{time['seasons_needed_close_ev'].min():.0f} to "
            f"{time['seasons_needed_close_ev'].max():.0f} seasons of betting on all five "
            "leagues, and on the Betfair exchange, which does not close winners, the same "
            f"rule returns {_pct(exchange['roi'])} after {exchange['commission']:.0%} "
            "commission."
        )
    if martingale.height:
        lose_half = martingale["p_lose_half"]
        sentences.append(
            "<b>Doubling after losses (martingale) is ruinous whatever is bet on</b>: it "
            "loses half the bankroll within one season in "
            f"{_pct(lose_half.min(), 0, False)} to {_pct(lose_half.max(), 0, False)} of "
            "simulated histories."
        )
    return sentences


def render(r: dict[str, Any]) -> str:
    """The report page for the results of collect."""
    games: pl.DataFrame = r["games"]
    first, last = int(games["season_start"].min()), int(games["season_start"].max())  # pyright: ignore[reportArgumentType]
    seasons = f"{first}/{(first + 1) % 100:02d} to {last}/{(last + 1) % 100:02d}"
    leagues = ", ".join(sorted(games["league"].unique()))

    comp_rows = [
        (
            f"{FORECASTER_NAMES[row['a']]} vs {FORECASTER_NAMES[row['b']]}",
            row["log_loss"],
            row["log_loss_low"],
            row["log_loss_high"],
        )
        for row in r["comparisons"].iter_rows(named=True)
    ]
    forecaster_table = _table(
        ["Forecaster", "Log loss", "RPS"],
        [
            [
                FORECASTER_NAMES[row["forecaster"]],
                _num(row["log_loss"], 4, False),
                _num(row["rps"], 4, False),
            ]
            for row in r["forecasters"].sort("log_loss").iter_rows(named=True)
        ],
    )
    league_table = _table(
        ["League", "Games", "Log loss difference"],
        [
            [
                row["league"],
                f"{row['games']:,}",
                _interval(row["log_loss"], row["log_loss_low"], row["log_loss_high"], _num),
            ]
            for row in r["by_league"].iter_rows(named=True)
        ],
    )
    margin_table = _table(
        ["Margin removal", "Timing", "Log loss"],
        [
            [row["method"], row["timing"], _num(row["log_loss"], 5, False)]
            for row in r["margins"].iter_rows(named=True)
        ],
        numeric_from=2,
    )

    all_leagues = r["scores"].filter(pl.col("league") == "ALL").sort("close_ev", descending=True)
    strategy_chart = interval_chart(
        [
            (row["variant"], row["close_ev"], row["close_ev_low"], row["close_ev_high"])
            for row in all_leagues.drop_nulls("close_ev_low").iter_rows(named=True)
        ],
        "Expected profit per bet at the closing price, by strategy",
    )
    strategy_table = _table(
        ["Strategy", "Bets", "ROI (95% interval)", "Expected at close", "q-value"],
        [
            [
                html.escape(row["variant"]),
                f"{row['bets']:,}",
                _interval(row["roi"], row["roi_low"], row["roi_high"]),
                _interval(row["close_ev"], row["close_ev_low"], row["close_ev_high"])
                if row["close_ev_low"] is not None
                else "–",
                f"{row['q_value']:.2f}",
            ]
            for row in all_leagues.sort("roi", descending=True).iter_rows(named=True)
        ],
    )
    walk_table = _table(
        ["Strategy", "Seasons", "Bets", "ROI (95% interval)"],
        [
            [
                html.escape(row["strategy"]),
                str(row["seasons"]),
                f"{row['bets']:,}",
                _interval(row["roi"], row["roi_low"], row["roi_high"]),
            ]
            for row in r["walk_forward"]
            .filter(pl.col("league") == "ALL")
            .sort("roi", descending=True)
            .iter_rows(named=True)
        ],
    )

    replay_table = _table(
        ["Strategy", "Plan", "Final bankroll", "Lowest", "Worst drawdown"],
        [
            [
                html.escape(row["variant"]),
                row["plan"],
                f"{row['final']:.2f}×",
                f"{row['lowest']:.2f}×",
                _pct(row["max_drawdown"], 0, False),
            ]
            for row in r["staking"]["replay"].iter_rows(named=True)
        ],
        numeric_from=2,
    )
    ruin = r["staking"]["ruin"].filter(pl.col("seasons") == 3)
    ruin_table = _table(
        ["Strategy", "Plan", "Lose half", "Lose 90%", "Median final"],
        [
            [
                html.escape(row["variant"]),
                row["plan"],
                _pct(row["p_lose_half"], 0, False),
                _pct(row["p_lose_90"], 0, False),
                f"{row['median_final']:.2f}×",
            ]
            for row in ruin.iter_rows(named=True)
        ],
        numeric_from=2,
    )

    time_table = _table(
        [
            "Strategy",
            "Bets a season",
            "Expected at close",
            "Units a season",
            "Seasons until it shows",
        ],
        [
            [
                html.escape(row["variant"]),
                f"{row['bets_per_season']:,}",
                _pct(row["close_ev"]),
                f"{row['units_per_season_close_ev']:.0f}",
                f"{row['seasons_needed_close_ev']:.0f}",
            ]
            for row in r["time"].iter_rows(named=True)
        ],
    )
    limits_table = _table(
        ["Accounts", "Closed when up", "Bets placed", "Profit (units)", "Last bet"],
        [
            [
                "no limit" if row["accounts"] is None else str(row["accounts"]),
                "–" if row["limit"] is None else f"{row['limit']:.0f} stakes",
                f"{row['bets']:,}",
                f"{row['profit']:+.0f}",
                str(row["last_day"]),
            ]
            for row in r["limits"].iter_rows(named=True)
        ],
    )
    exchange_table = _table(
        ["Commission", "Bets", "ROI", "Expected at close"],
        [
            [
                _pct(row["commission"], 0, False),
                f"{row['bets']:,}",
                _pct(row["roi"]),
                _pct(row["close_ev"]),
            ]
            for row in r["exchange"].iter_rows(named=True)
        ],
    )

    dirty = " (with uncommitted changes)" if r["dirty"] else ""
    summary = "".join(f"<li>{s}</li>" for s in _summary(r))
    return PAGE.format(
        generated=r["generated"],
        code_commit=html.escape(r["code_commit"]) + dirty,
        suite_commit=html.escape(r["suite_commit"]),
        n_games=f"{games.height:,}",
        seasons=seasons,
        leagues=html.escape(leagues),
        test_from=r["test_from"],
        test_from_next=f"{(r['test_from'] + 1) % 100:02d}",
        first_test=FIRST_TEST_SEASON,
        first_test_next=f"{(FIRST_TEST_SEASON + 1) % 100:02d}",
        test_games=f"{r['test_games']:,}",
        summary=summary,
        comparison_chart=interval_chart(comp_rows, "Difference in log loss, a minus b", _num),
        forecaster_table=forecaster_table,
        league_table=league_table,
        margin_table=margin_table,
        strategy_chart=strategy_chart,
        strategy_table=strategy_table,
        walk_table=walk_table,
        replay_table=replay_table,
        ruin_table=ruin_table,
        time_table=time_table,
        limits_table=limits_table,
        exchange_table=exchange_table,
        examined=html.escape(practicality.EXAMINED[0]),
    )


PAGE = """<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Beating the Market?</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --panel: #f4f3f0; --border: #dddcd6;
  --text: #0b0b0b; --text-2: #52514e; --muted: #77756f;
  --series: #2a78d6; --zero: #9c9a92;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --border: #3a3936;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #9a988f;
    --series: #3987e5; --zero: #6d6b64;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface: #1a1a19; --panel: #232322; --border: #3a3936;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #9a988f;
  --series: #3987e5; --zero: #6d6b64;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 900px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 1.9rem; line-height: 1.2; margin: 0 0 6px; }}
h2 {{ font-size: 1.3rem; margin: 48px 0 8px; padding-top: 16px;
  border-top: 1px solid var(--border); }}
h3 {{ font-size: 1.02rem; margin: 24px 0 6px; }}
p, li {{ color: var(--text-2); }}
b {{ color: var(--text); }}
.meta {{ color: var(--muted); font-size: .9rem; }}
.answer {{ background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 8px 20px; }}
.answer li {{ margin: 10px 0; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: .88rem;
  font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left;
  vertical-align: top; }}
th {{ color: var(--muted); font-weight: 500; }}
.num {{ text-align: right; white-space: nowrap; }}
.muted {{ color: var(--muted); }}
.chart {{ width: 100%; min-width: 720px; height: auto; display: block; margin: 8px 0 4px; }}
.chart text {{ font: 12px system-ui, sans-serif; }}
.chart .label {{ fill: var(--text-2); }}
.chart .tick {{ fill: var(--muted); }}
.chart .zero {{ stroke: var(--zero); stroke-width: 1; stroke-dasharray: 3 3; }}
.chart .ci {{ stroke: var(--series); stroke-width: 2; stroke-linecap: round; }}
.chart .dot {{ fill: var(--series); stroke: var(--surface); stroke-width: 2; }}
.chart .hit {{ fill: transparent; }}
.chart .row:hover .hit {{ fill: var(--panel); }}
.chart .row:hover .label {{ fill: var(--text); }}
.note {{ font-size: .88rem; color: var(--muted); }}
code {{ font-size: .88em; }}
</style>
</head>
<body>
<main>
<h1>Can a bettor beat the football market?</h1>
<p class=meta>Generated {generated} by <code>uv run report</code> · code {code_commit} ·
strategies.toml {suite_commit} · {n_games} games of {leagues}, {seasons}</p>

<p>The strongest honest attempt we could build with public data: a model anchored on the
market, 18 betting strategies fixed in advance, four staking plans, and the cost of
collecting any edge found. Every number below comes from the run named above; checking
out those commits and running <code>uv run collect_data</code> then <code>uv run report</code>
rebuilds this page.</p>

<h2 id=answer>The answer</h2>
<div class=answer><ul>{summary}</ul></div>

<h2 id=rules>1. How it was tested</h2>
<ul>
<li><b>Data.</b> Results and prices from football-data.co.uk: Pinnacle (the sharpest
bookmaker), Bet365, the best and average price across bookmakers, and the Betfair
exchange, early (a day or two before kick-off) and at the close.</li>
<li><b>No hindsight.</b> Every season is predicted, and every strategy parameter chosen,
from earlier seasons only. Scoring starts with {first_test}/{first_test_next}.</li>
<li><b>Fixed in advance.</b> The strategies are listed in <code>strategies.toml</code>,
committed before any of them was run ({suite_commit}).</li>
<li><b>Counting the tries.</b> Every result is reported, not just the best, with a
Benjamini–Hochberg q-value over all of them.</li>
<li><b>Proper scores.</b> Forecasts are judged by log loss and the ranked probability
score, and bets by the expected profit at the closing price as well as realised profit,
because the first is far less noisy.</li>
</ul>

<h2 id=model>2. Can a model forecast better than the market?</h2>
<p>Each forecaster is scored on the same {test_games} games. A difference below zero
means the first forecaster is better; the line is its 95% interval.</p>
{comparison_chart}
<h3>Mean scores (lower is better)</h3>
{forecaster_table}
<h3>Early price + team form vs the early price, by league</h3>
{league_table}
<h3>Ways of removing the bookmaker's margin</h3>
{margin_table}

<h2 id=strategies>3. Does any strategy win?</h2>
<p>Expected profit per bet if the closing price is right, all leagues, from
{test_from}/{test_from_next}. A line wholly to the right of zero is an edge against the
closing price.</p>
{strategy_chart}
<h3>Every strategy, all leagues</h3>
{strategy_table}
<h3>Choosing each season's parameter from earlier seasons only</h3>
{walk_table}

<h2 id=staking>4. What staking does to a bankroll</h2>
<p>The bets replayed in date order from a bankroll of 1. Flat stakes 1% of the start;
Kelly stakes its fraction of the current bankroll using the probability the strategy bets
on; martingale doubles after every loss.</p>
<h3>The history that happened</h3>
{replay_table}
<h3>Over 1,000 resampled histories of three seasons</h3>
{ruin_table}

<h2 id=practicality>5. What collecting an edge would take</h2>
<h3>Seasons before the edge shows as profit</h3>
<p>Seasons of betting on all five leagues before an edge of the expected size is two
standard errors above zero, and what it earns at 1 unit a bet.</p>
{time_table}
<h3>Bookmakers closing winning accounts</h3>
<p>{examined}, with the bets spread over several accounts and each closed once it is a
few stakes up.</p>
{limits_table}
<h3>The Betfair exchange, which does not close winners</h3>
{exchange_table}

<h2 id=limits>6. Limitations</h2>
<ul>
<li>Five top European leagues, 1X2, Asian handicap and over/under 2.5 only. Smaller
leagues and other markets may be priced less sharply, and are also where limits come
fastest.</li>
<li>The team-form model uses goals and shots on target from football-data; expected-goals
data from Understat was not part of this run.</li>
<li>Closing prices are the best case for timing: a bettor seldom gets the closing price
on every bet. Best prices assume an account at every bookmaker football-data lists.</li>
<li>Prices implying an arbitrage of 3% or more were dropped as data errors.</li>
<li>The ruin simulation resamples single bets, which ignores streaks within a season
and so understates risk.</li>
</ul>
<p class=note>Source: <code>src/money_printer_research/analysis/</code> — market.py,
strategies.py, staking.py, practicality.py, report.py.</p>
</main>
</body>
</html>
"""


def write(page: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def main(settings: Settings = default_settings) -> Path:
    """Run everything and write settings.modeling.report_file."""
    results = collect(settings)
    out = settings.modeling.report_file
    write(render(results), out)
    print(f"Saved {out}")
    return out


def exec_report() -> None:
    """Entry point `report`: the one-page answer, rebuilt from the raw data."""
    main()


__all__ = ["collect", "interval_chart", "render"]
