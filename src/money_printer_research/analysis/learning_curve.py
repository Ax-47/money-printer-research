"""How the H/D/A models learn as they see more games: the learning curve.

Each model is fitted on the most recent n games before TEST_FROM, for n doubling from
SMALLEST up to all of them, and scored on the same held-out games from TEST_FROM on.
Two numbers per fit:

    train_log_loss  on the games it was fitted on
    test_log_loss   on the held-out games, with its standard error

The gap between them is how much the model memorises; where the test curve flattens,
more games of the same kind stop helping. Two lines give the curves a scale: the
log loss of the training games' H/D/A shares alone (a model that knows nothing about
the teams), and of the bookmaker's margin-free probabilities on the same games (the
market's own forecast, analysis.backtest's prices). When there are odds, the held-out
games are those with a price, so every line is scored on the same games.

`uv run learning_curve` prints the table and writes settings.modeling.learning_curve_file,
a page with the curves; `uv run modeling` does the same after saving the model.
"""

import html
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from money_printer_research.analysis.backtest import OUTCOMES, attach_odds, read_odds
from money_printer_research.analysis.feature_importance import (
    CLASSES,
    SELECTED,
    TARGET,
    TEST_FROM,
    columns,
    load,
)
from money_printer_research.analysis.modeling import MODELS, Model
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

SMALLEST = 250
"""The fewest training games: below this, a season of one league, the fits are noise."""

SAVED_MODEL = "logistic"
"""The model `modeling` saves, whose train and test curves the page shows together."""

GAME = ["league", "season_start", "home_team", "away_team"]


def training_sizes(available: int, smallest: int = SMALLEST) -> list[int]:
    """smallest, 2 * smallest, ... while below `available`, then `available` itself."""
    sizes, n = [], smallest
    while n < available:
        sizes.append(n)
        n *= 2
    return [*sizes, available]


def game_log_loss(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """-log of the probability each game's outcome was given (columns in CLASSES order)."""
    given = proba[np.arange(len(y)), [CLASSES.index(c) for c in y]]
    return -np.log(np.clip(given, 1e-15, 1))


def held_out_games(df: pl.DataFrame, odds: pl.DataFrame | None) -> pl.DataFrame:
    """The games from TEST_FROM on, with odds_H/D/A when `odds` is given.

    Games without a price are then dropped, so the market is scored on the same games.
    """
    test = df.filter(pl.col("season_start") >= TEST_FROM)
    if odds is None:
        return test
    priced = attach_odds(test.select("game_id", "date", *GAME), odds)
    return test.join(priced.select("game_id", *(f"odds_{k}" for k in OUTCOMES)), on="game_id")


def references(train_pool: pl.DataFrame, test: pl.DataFrame) -> dict[str, float]:
    """Test log loss of the training games' outcome shares and, with odds, of the market."""
    y = test[TARGET].to_numpy()
    shares = train_pool[TARGET].value_counts(normalize=True)
    share = dict(zip(shares[TARGET], shares["proportion"], strict=True))
    base = np.tile([share.get(c, 0.0) for c in CLASSES], (len(y), 1))
    out = {"base_rates": float(game_log_loss(y, base).mean())}
    if "odds_H" in test.columns:
        inverse = 1 / test.select(f"odds_{c}" for c in CLASSES).to_numpy()
        market = inverse / inverse.sum(axis=1, keepdims=True)
        out["market"] = float(game_log_loss(y, market).mean())
    return out


def learning_curve(
    df: pl.DataFrame,
    test: pl.DataFrame,
    models: dict[str, Model] = MODELS,
    groups: list[str] = SELECTED,
    sizes: list[int] | None = None,
) -> pl.DataFrame:
    """model, games, first_season, train_log_loss, test_log_loss, test_se: one row per fit.

    The training games are the most recent `games` before TEST_FROM; each fit predicts its
    own training games and `test` in one call, so it is fitted once.
    """
    pool = df.filter(pl.col("season_start") < TEST_FROM).sort("date", "game_id")
    cols = columns(groups)
    y_test = test[TARGET].to_numpy()
    rows = []
    for n in sizes or training_sizes(pool.height):
        train = pool.tail(n)
        both = pl.concat([train, test.select(train.columns)])
        for name, model in models.items():
            proba = model(train, both, cols)
            train_loss = game_log_loss(train[TARGET].to_numpy(), proba[:n])
            test_loss = game_log_loss(y_test, proba[n:])
            rows.append(
                {
                    "model": name,
                    "games": n,
                    "first_season": int(train["season_start"].min()),  # pyright: ignore[reportArgumentType]
                    "train_log_loss": float(train_loss.mean()),
                    "test_log_loss": float(test_loss.mean()),
                    "test_se": float(test_loss.std(ddof=1) / math.sqrt(len(test_loss))),
                }
            )
    return pl.DataFrame(rows)


# --- The page --------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
"""Categorical slots 1-4, light and dark; a model keeps its colour on every chart."""

MODEL_TH = {
    "logistic": "logistic regression",
    "logistic + sigmoid": "logistic + ปรับเทียบ sigmoid",
    "boosted trees": "boosted trees",
    "poisson goals": "โมเดลประตู Poisson",
}


def _tr(en: str, th: str) -> str:
    """Both languages; the page shows the one its language button picks."""
    return f"<span class=en>{html.escape(en)}</span><span class=th>{html.escape(th)}</span>"


def _label(en: str, th: str, after: str = "") -> str:
    en, th = html.escape(en), html.escape(th)
    return f"<tspan class=en>{en}</tspan><tspan class=th>{th}</tspan>{after}"


def spread(ys: list[float], gap: float = 15) -> list[float]:
    """The label heights `ys`, moved at least `gap` apart and kept centred on their lines.

    Labels that would collide form a block, evenly spaced and centred on their mean
    height; blocks that then collide merge, until none do.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    blocks = [[i] for i in order]

    def top(block: list[int]) -> float:
        return sum(ys[i] for i in block) / len(block) - (len(block) - 1) * gap / 2

    merged = True
    while merged:
        merged = False
        for k in range(len(blocks) - 1):
            if top(blocks[k]) + len(blocks[k]) * gap > top(blocks[k + 1]):
                blocks[k : k + 2] = [blocks[k] + blocks[k + 1]]
                merged = True
                break
    out = [0.0] * len(ys)
    for block in blocks:
        for j, i in enumerate(block):
            out[i] = top(block) + j * gap
    return out


def line_chart(
    lines: list[dict[str, Any]], refs: list[dict[str, Any]], width: int = 860, height: int = 380
) -> str:
    """SVG: x = training games (log scale), y = log loss.

    lines: {label_en, label_th, colour (a CSS var), dashed, points: [(games, loss, tip)]}.
    refs: horizontal lines {label_en, label_th, value, css_class}.
    """
    left, right, top, bottom = 56, 190, 16, 46
    xs = [p[0] for line in lines for p in line["points"]]
    ys = [p[1] for line in lines for p in line["points"]] + [r["value"] for r in refs]
    x_lo, x_hi = math.log2(min(xs)), math.log2(max(xs))
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.06 or 0.01
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def x(v: float) -> float:
        span = x_hi - x_lo or 1
        return left + (math.log2(v) - x_lo) / span * (width - left - right)

    def y(v: float) -> float:
        return top + (y_hi - v) / (y_hi - y_lo) * (height - top - bottom)

    parts = [
        f'<svg class=chart viewBox="0 0 {width} {height}" role=img '
        'aria-label="learning curve: log loss against training games">'
    ]
    for tick in np.linspace(y_lo + pad, y_hi - pad, 5):
        parts.append(
            f'<line class=grid x1={left} x2={width - right} y1="{y(tick):.1f}" '
            f'y2="{y(tick):.1f}"/><text class=tick x={left - 6} y="{y(tick) + 4:.1f}" '
            f"text-anchor=end>{tick:.3f}</text>"
        )
    for games in sorted(set(xs)):
        parts.append(
            f'<text class=tick x="{x(games):.1f}" y={height - bottom + 18} '
            f"text-anchor=middle>{games:,}</text>"
        )
    parts.append(
        f'<text class=tick x="{(left + width - right) / 2:.0f}" y={height - 6} '
        "text-anchor=middle><tspan class=en>games the model was fitted on (log scale)</tspan>"
        "<tspan class=th>จำนวนนัดที่ใช้เทรน (สเกล log)</tspan></text>"
    )
    labels: list[tuple[float, str]] = []
    """(y, inner text) of each right-hand label, placed once all are known."""
    for ref in refs:
        yy = y(ref["value"])
        parts.append(
            f'<line class="ref {ref["css_class"]}" x1={left} x2={width - right} '
            f'y1="{yy:.1f}" y2="{yy:.1f}"/>'
        )
        labels.append((yy, _label(ref["label_en"], ref["label_th"], f" {ref['value']:.4f}")))
    for line in lines:
        points = " ".join(f"{x(g):.1f},{y(v):.1f}" for g, v, _ in line["points"])
        dash = " stroke-dasharray='5 4'" if line["dashed"] else ""
        parts.append(f'<polyline class=series points="{points}" stroke="{line["colour"]}"{dash}/>')
        for g, v, tip in line["points"]:
            parts.append(
                f'<g class=pt><circle class=hit cx="{x(g):.1f}" cy="{y(v):.1f}" r="12" />'
                f'<circle class=dot cx="{x(g):.1f}" cy="{y(v):.1f}" r=4.5 '
                f'fill="{line["colour"]}"/><title>{html.escape(tip)}</title></g>'
            )
        labels.append((y(line["points"][-1][1]), _label(line["label_en"], line["label_th"])))
    for yy, inner in zip(spread([yy for yy, _ in labels]), (i for _, i in labels), strict=True):
        parts.append(f'<text class=label x={width - right + 8} y="{yy + 4:.1f}">{inner}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(curve: pl.DataFrame, refs: dict[str, float], test_games: int, made: str) -> str:
    """The learning-curve page for learning_curve's table and references' lines."""
    models: list[str] = list(dict.fromkeys(curve["model"].to_list()))
    colour = {m: f"var(--s{i + 1})" for i, m in enumerate(models)}
    ref_lines = [
        {
            "label_en": "outcome shares only",
            "label_th": "สัดส่วนผลอย่างเดียว",
            "value": refs["base_rates"],
            "css_class": "base",
        }
    ]
    if "market" in refs:
        ref_lines.append(
            {
                "label_en": "bookmaker",
                "label_th": "เจ้ามือ",
                "value": refs["market"],
                "css_class": "market",
            }
        )

    def points(rows: pl.DataFrame, key: str) -> list[tuple[int, float, str]]:
        return [
            (r["games"], r[key], f"{r['model']}, {r['games']:,} games: {key} {r[key]:.4f}")
            for r in rows.iter_rows(named=True)
        ]

    test_lines = [
        {
            "label_en": m,
            "label_th": MODEL_TH.get(m, m),
            "colour": colour[m],
            "dashed": False,
            "points": points(curve.filter(pl.col("model") == m), "test_log_loss"),
        }
        for m in models
    ]
    saved = curve.filter(pl.col("model") == SAVED_MODEL)
    gap_lines = [
        {
            "label_en": "held-out games",
            "label_th": "นัดที่ไม่ได้ใช้เทรน",
            "colour": colour.get(SAVED_MODEL, "var(--s1)"),
            "dashed": False,
            "points": points(saved, "test_log_loss"),
        },
        {
            "label_en": "training games",
            "label_th": "นัดที่ใช้เทรน",
            "colour": colour.get(SAVED_MODEL, "var(--s1)"),
            "dashed": True,
            "points": points(saved, "train_log_loss"),
        },
    ]

    best = curve.sort("test_log_loss").row(0, named=True)
    smallest, largest = curve["games"].min(), curve["games"].max()
    first = saved.filter(pl.col("games") == smallest).row(0, named=True)
    last = saved.filter(pl.col("games") == largest).row(0, named=True)
    half = saved.filter(pl.col("games") < largest).sort("games").tail(1)
    gain_last = (half["test_log_loss"].item() - last["test_log_loss"]) if half.height else 0.0
    findings = [
        _tr(
            f"The best fit is {best['model']} on {best['games']:,} games: held-out log loss "
            f"{best['test_log_loss']:.4f} (± {best['test_se']:.4f}).",
            f"ดีที่สุดคือ {MODEL_TH.get(best['model'], best['model'])} ที่เทรนด้วย "
            f"{best['games']:,} นัด: log loss บนนัดที่ไม่ได้ใช้เทรน {best['test_log_loss']:.4f} "
            f"(± {best['test_se']:.4f})",
        ),
        _tr(
            f"The {SAVED_MODEL} model goes from {first['test_log_loss']:.4f} on "
            f"{first['games']:,} games to {last['test_log_loss']:.4f} on {last['games']:,}; "
            f"the last doubling of the data gains {gain_last:+.4f}.",
            f"โมเดล {SAVED_MODEL} ดีขึ้นจาก {first['test_log_loss']:.4f} ที่ {first['games']:,} "
            f"นัด เป็น {last['test_log_loss']:.4f} ที่ {last['games']:,} นัด "
            f"ข้อมูลที่เพิ่มรอบสุดท้ายช่วยได้ {gain_last:+.4f}",
        ),
        _tr(
            f"On all its games it scores {last['train_log_loss']:.4f} on the games it was "
            f"fitted on and {last['test_log_loss']:.4f} on the held-out ones: a gap of "
            f"{last['test_log_loss'] - last['train_log_loss']:+.4f}.",
            f"เมื่อเทรนด้วยทุกนัด ได้ {last['train_log_loss']:.4f} บนนัดที่ใช้เทรน และ "
            f"{last['test_log_loss']:.4f} บนนัดที่ไม่ได้ใช้เทรน ห่างกัน "
            f"{last['test_log_loss'] - last['train_log_loss']:+.4f}",
        ),
    ]
    if "market" in refs:
        behind = best["test_log_loss"] - refs["market"]
        findings.append(
            _tr(
                f"The bookmaker's prices score {refs['market']:.4f} on the same games: the "
                f"best model is {behind:+.4f} behind them.",
                f"ราคาเจ้ามือได้ {refs['market']:.4f} บนนัดชุดเดียวกัน โมเดลที่ดีที่สุดยังตามหลัง {behind:+.4f}",
            )
        )

    table_rows = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td class=num>{r['games']:,}</td>"
        f"<td class=num>{r['first_season']}</td><td class=num>{r['train_log_loss']:.4f}</td>"
        f"<td class=num>{r['test_log_loss']:.4f}</td><td class=num>{r['test_se']:.4f}</td></tr>"
        for r in curve.sort("model", "games").iter_rows(named=True)
    )
    legend = "".join(
        f'<span><i style="background:{colour[m]}"></i>{_tr(m, MODEL_TH.get(m, m))}</span>'
        for m in models
    )
    return PAGE.format(
        made=made,
        test_from=TEST_FROM,
        test_games=f"{test_games:,}",
        findings="".join(f"<li>{f}</li>" for f in findings),
        legend=legend,
        test_chart=line_chart(test_lines, ref_lines),
        gap_chart=line_chart(gap_lines, []),
        rows=table_rows,
        s="".join(f"--s{i + 1}: {c}; " for i, c in enumerate(SERIES)),
        s_dark="".join(f"--s{i + 1}: {c}; " for i, c in enumerate(SERIES_DARK)),
    )


PAGE = """<!doctype html>
<html lang=th>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Learning Curve</title>
<style>
:root {{
  color-scheme: light; {s}
  --surface: #fcfcfb; --panel: #f4f3f0; --border: #dddcd6; --grid: #ebeae6;
  --text: #0b0b0b; --text-2: #52514e; --muted: #77756f;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark; {s_dark}
    --surface: #1a1a19; --panel: #232322; --border: #3a3936; --grid: #2c2c2a;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #9a988f;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark; {s_dark}
  --surface: #1a1a19; --panel: #232322; --border: #3a3936; --grid: #2c2c2a;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #9a988f;
}}
html[lang=th] .en, html[lang=en] .th {{ display: none; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 900px; margin: 0 auto; padding: 32px 16px 64px; }}
header {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
h1 {{ font-size: 1.7rem; line-height: 1.2; margin: 0 0 6px; }}
h2 {{ font-size: 1.2rem; margin: 40px 0 6px; }}
p, li {{ color: var(--text-2); }}
.meta {{ color: var(--muted); font-size: .9rem; }}
button {{ font: inherit; font-size: .85rem; padding: 4px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--panel); color: var(--text);
  cursor: pointer; }}
.findings {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 20px 8px 36px; }}
.findings li {{ margin: 8px 0; }}
.scroll {{ overflow-x: auto; }}
.chart {{ width: 100%; min-width: 640px; height: auto; display: block; }}
.chart text {{ font: 12px system-ui, sans-serif; }}
.chart .tick {{ fill: var(--muted); }}
.chart .label {{ fill: var(--text-2); }}
.chart .grid {{ stroke: var(--grid); }}
.chart .series {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.chart .dot {{ stroke: var(--surface); stroke-width: 2; }}
.chart .hit {{ fill: transparent; }}
.chart .pt:hover .dot {{ r: 6; }}
.chart .ref {{ stroke-width: 1.5; stroke-dasharray: 2 4; }}
.chart .ref.base {{ stroke: var(--muted); }}
.chart .ref.market {{ stroke: var(--text); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: .85rem; color: var(--text-2); }}
.legend i {{ display: inline-block; width: 16px; height: 3px; margin-right: 6px;
  vertical-align: middle; border-radius: 2px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .88rem;
  font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
th {{ color: var(--muted); font-weight: 500; }}
.num {{ text-align: right; }}
.note {{ font-size: .85rem; color: var(--muted); }}
</style>
</head>
<body>
<main>
<header>
<div>
<h1><span class=en>How the models learn</span><span class=th>โมเดลเรียนรู้อย่างไร</span></h1>
<p class=meta><span class=en>Generated {made} · held-out games from {test_from}:
{test_games}</span><span class=th>สร้างเมื่อ {made} · นัดทดสอบตั้งแต่ฤดูกาล {test_from}:
{test_games} นัด</span></p>
</div>
<button id=lang type=button></button>
</header>

<p><span class=en>Each model is fitted on the most recent games before {test_from}, from a
few hundred up to all of them, and scored on the same held-out games. Log loss: lower is
better.</span><span class=th>แต่ละโมเดลเทรนด้วยนัดล่าสุดก่อนฤดูกาล {test_from} ตั้งแต่ไม่กี่ร้อยนัด
จนถึงทั้งหมด แล้ววัดผลบนนัดทดสอบชุดเดียวกัน ค่า log loss ยิ่งต่ำยิ่งดี</span></p>

<ul class=findings>{findings}</ul>

<h2><span class=en>Held-out log loss against training games</span><span class=th>log loss
บนนัดทดสอบ เทียบกับจำนวนนัดที่ใช้เทรน</span></h2>
<p><span class=en>Where a curve flattens, more games stop helping. The dotted lines are the
scale: what the outcome shares alone score, and what the bookmaker's prices
score.</span><span class=th>ถ้าเส้นเริ่มแบน แปลว่าเพิ่มข้อมูลก็ไม่ช่วยแล้ว เส้นประคือเกณฑ์เทียบ:
ทายจากสัดส่วนผลอย่างเดียว และราคาของเจ้ามือ</span></p>
<div class=legend>{legend}</div>
<div class=scroll>{test_chart}</div>

<h2><span class=en>The saved model: training games against held-out games</span><span
class=th>โมเดลที่บันทึกไว้: นัดที่ใช้เทรน เทียบกับนัดทดสอบ</span></h2>
<p><span class=en>The dashed line is the score on the games the model was fitted on, the
solid one on the held-out games. A wide gap is memorising; lines that meet mean the
model has learnt what these features can tell.</span><span class=th>เส้นประคือคะแนนบนนัดที่ใช้เทรน
เส้นทึบคือบนนัดทดสอบ ถ้าห่างกันมากแปลว่าโมเดลจำข้อมูล (overfit) ถ้าสองเส้นมาชนกัน
แปลว่าโมเดลเรียนรู้จาก feature ชุดนี้ได้เต็มที่แล้ว</span></p>
<div class=scroll>{gap_chart}</div>

<h2><span class=en>Every fit</span><span class=th>ผลทุกครั้งที่เทรน</span></h2>
<div class=scroll><table>
<thead><tr><th><span class=en>Model</span><span class=th>โมเดล</span></th>
<th class=num><span class=en>Training games</span><span class=th>นัดที่ใช้เทรน</span></th>
<th class=num><span class=en>From season</span><span class=th>ตั้งแต่ฤดูกาล</span></th>
<th class=num><span class=en>Train log loss</span><span class=th>log loss นัดที่ใช้เทรน</span></th>
<th class=num><span class=en>Held-out log loss</span><span class=th>log loss นัดทดสอบ</span></th>
<th class=num><span class=en>± SE</span><span class=th>± SE</span></th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class=note><span class=en>Rebuild with <code>uv run learning_curve</code> (also run by
<code>uv run modeling</code>).</span><span class=th>สร้างใหม่ด้วย <code>uv run learning_curve</code>
(<code>uv run modeling</code> ก็สร้างให้ด้วย)</span></p>
</main>
<script>
  const root = document.documentElement, button = document.getElementById("lang");
  function show(lang) {{
    root.lang = lang;
    button.textContent = lang === "th" ? "EN" : "ไทย";
    try {{ localStorage.setItem("lang", lang); }} catch {{}}
  }}
  let saved = "th";
  try {{ saved = localStorage.getItem("lang") || "th"; }} catch {{}}
  show(saved === "en" ? "en" : "th");
  button.addEventListener("click", () => show(root.lang === "th" ? "en" : "th"));
</script>
</body>
</html>
"""


def run(settings: Settings = default_settings) -> tuple[pl.DataFrame, dict[str, float], int]:
    """The learning curve, its reference lines and the number of held-out games."""
    df = load(settings.clean.out_dir / "model_frame.csv")
    odds_dir = settings.modeling.odds_dir
    odds = read_odds(odds_dir) if any(odds_dir.glob("*/*.csv")) else None
    test = held_out_games(df, odds)
    curve = learning_curve(df, test)
    refs = references(df.filter(pl.col("season_start") < TEST_FROM), test)
    return curve, refs, test.height


def write(page: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def main(settings: Settings = default_settings) -> Path:
    """Print the learning curve and write its page to settings.modeling.learning_curve_file."""
    curve, refs, test_games = run(settings)
    with pl.Config(float_precision=4, tbl_rows=60):
        print(f"Learning curve, scored on {test_games:,} held-out games from {TEST_FROM}:")
        print(curve)
    print(", ".join(f"{name} {value:.4f}" for name, value in refs.items()))
    path = settings.modeling.learning_curve_file
    write(render(curve, refs, test_games, date.today().isoformat()), path)
    print(f"Saved {path}")
    return path


def exec_learning_curve() -> None:
    """Entry point `learning_curve`: how the models learn as they see more games."""
    main()


__all__ = ["learning_curve", "references", "render", "training_sizes"]
