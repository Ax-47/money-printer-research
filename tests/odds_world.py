"""football-data.co.uk files for one league over several seasons, as collect_data saves them.

Six teams play a double round robin each season. Every game has the prices the
pre-registered suite (strategies.toml) bets at: Pinnacle, Bet365, the best and the average
price, early and closing, for 1X2, Asian handicap and over/under 2.5, plus the Betfair
exchange. Results are drawn from the margin-free Pinnacle probabilities, so no strategy
has an edge by construction.
"""

import itertools
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

LEAGUE = "ENG-Premier League"
TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham", "Leeds", "Stoke"]
SEASONS = [2013, 2014, 2015, 2016, 2017]


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def write_odds(odds_dir: Path, seed: int = 0) -> Path:
    """Write one CSV per season under odds_dir/LEAGUE and return odds_dir."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, rng.normal(0, 0.4, len(TEAMS)), strict=True)}
    (odds_dir / LEAGUE).mkdir(parents=True, exist_ok=True)
    for year in SEASONS:
        rows = []
        for i, (home, away) in enumerate(itertools.permutations(TEAMS, 2)):
            gap = strength[home] - strength[away] + 0.3
            p_home = 1 / (1 + np.exp(-gap)) * 0.75
            p = np.array([p_home, 0.25, 0.75 - p_home])
            outcome = rng.choice(3, p=p)
            hg, ag = [(2, 0), (1, 1), (0, 1)][outcome]
            hg += int(rng.integers(0, 2))
            ag += int(rng.integers(0, 2)) if outcome != 1 else hg - ag
            early, close = 0.96 / p, 0.97 / (p * rng.uniform(0.97, 1.03, 3))
            best = early * rng.uniform(1.0, 1.06, 3)
            p_over = 0.52
            day = date(year, 8, 10) + timedelta(days=3 * i)
            row = {
                "Div": "E0",
                "Date": day.strftime("%d/%m/%Y"),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "HST": int(rng.integers(1, 9)),
                "AST": int(rng.integers(0, 7)),
                "AHh": "-0.5",
                "AHCh": "-0.5",
            }
            for k, side in enumerate("HDA"):
                row[f"PS{side}"] = _fmt(early[k])
                row[f"PSC{side}"] = _fmt(close[k])
                row[f"B365{side}"] = _fmt(early[k] * 0.98)
                row[f"Max{side}"] = _fmt(best[k])
                row[f"MaxC{side}"] = _fmt(close[k] * rng.uniform(1.0, 1.06))
                row[f"Avg{side}"] = _fmt(early[k] * 0.97)
                row[f"BFE{side}"] = _fmt(early[k] * 1.02)
            for prefix in ("P", "Max"):
                bump = rng.uniform(1.0, 1.05) if prefix == "Max" else 1.0
                row[f"{prefix}>2.5"] = _fmt(0.97 / p_over * bump)
                row[f"{prefix}<2.5"] = _fmt(0.97 / (1 - p_over) * bump)
                row[f"{prefix}AHH"] = _fmt(1.93 * bump)
                row[f"{prefix}AHA"] = _fmt(1.95 * bump)
            row["PC>2.5"], row["PC<2.5"] = _fmt(0.97 / p_over), _fmt(0.97 / (1 - p_over))
            row["PCAHH"], row["PCAHA"] = "1.94", "1.94"
            rows.append(row)
        name = f"{year}-{str(year + 1)[-2:]}.csv"
        pl.DataFrame(rows).write_csv(odds_dir / LEAGUE / name)
    return odds_dir
