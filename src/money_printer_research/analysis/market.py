"""Can a model beat the betting market's own probabilities (issue #38)?

The bar is the bookmaker's price with its margin removed. Pinnacle's prices are used
because they are the sharpest widely published ones, at two moments:

    early  a day or two before kick-off: the price a bettor can act on
    close  at kick-off: everything anyone knew by then

Every forecaster is scored on the same games, each season predicted only from seasons
before it:

    early, close      the market's margin-free probabilities (power method, see
                      margin_free), taken as they are
    market_refit      the early market recalibrated by a logistic regression: does the
                      market have a bias a model could exploit?
    market_and_form   the early market plus team form (Elo, recent goals and shots on
                      target): does what we know about the teams add anything?
    form_only         team form without the market: how far a model gets on its own

Per game, log loss and the ranked probability score (RPS); lower is better for both.
paired_differences compares two forecasters game by game, so the noise that games add
to both cancels, and gives a 95% interval for the mean difference. A model "beats" the
market only if that interval lies entirely below zero.

Everything here uses football-data.co.uk alone (prices, results, shots on target), so
it runs without the Understat pipeline.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from money_printer_research.analysis.odds import GAME_KEY, one_price, read_prices, read_results
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings

OUTCOMES = ["H", "D", "A"]
FORECASTERS = ["early", "close", "market_refit", "market_and_form", "form_only"]
FIRST_TEST_SEASON = 2016
"""Seasons before this only train: form needs a season to settle, models need history."""


# --- Removing the bookmaker's margin -------------------------------------------------


def margin_free(odds: np.ndarray, method: str = "power") -> np.ndarray:
    """Probabilities from decimal odds (one row per game), with the margin removed.

    The implied probabilities 1/odds sum to more than 1; the excess is the margin.
        proportional  divide each by their sum
        power         raise each to the power k that makes them sum to 1; takes more
                      margin off long shots, where bookmakers put more of it
        shin          Shin's (1993) model of a bookmaker facing some insiders
    """
    implied = 1 / odds
    if method == "proportional":
        return implied / implied.sum(axis=1, keepdims=True)
    solve = {"power": _power_row, "shin": _shin_row}[method]
    return np.array([solve(row) for row in implied])


def _power_row(implied: np.ndarray) -> np.ndarray:
    k = brentq(lambda k: (implied**k).sum() - 1, 0.5, 3)
    return implied**k


def _shin_row(implied: np.ndarray) -> np.ndarray:
    total = implied.sum()

    def probabilities(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4 * (1 - z) * implied**2 / total) - z) / (2 * (1 - z))

    z = brentq(lambda z: probabilities(z).sum() - 1, 0, 0.5)
    return probabilities(z)


# --- The games, with what was known before each kick-off -----------------------------


def read_games(odds_dir: Path) -> pl.DataFrame:
    """One row per game with Pinnacle's early and closing 1X2 prices and the result.

    Columns: the game key, result (H/D/A), early_H/D/A, close_H/D/A, the score and
    shots on target. Sorted by day, the order team_form replays them in.
    """
    prices = read_prices(odds_dir)
    early = one_price(prices, "1x2", [("pinnacle", "early")])
    close = one_price(prices, "1x2", [("pinnacle", "close")])
    games = (
        read_results(odds_dir)
        .join(early.rename({f"odds_{k}": f"early_{k}" for k in OUTCOMES}), on=GAME_KEY)
        .join(close.rename({f"odds_{k}": f"close_{k}" for k in OUTCOMES}), on=GAME_KEY)
    )
    goals = pl.col("home_goals") - pl.col("away_goals")
    result = pl.when(goals > 0).then(pl.lit("H")).when(goals == 0).then(pl.lit("D"))
    return games.with_columns(result=result.otherwise(pl.lit("A"))).sort(
        "day", "league", "fd_home"
    )


FORM_COLUMNS = [
    "elo_diff",
    "goals_for_diff",
    "goals_against_diff",
    "sot_for_diff",
    "sot_against_diff",
]


@dataclass
class _Form:
    """A team's running form: Elo rating and exponentially weighted averages."""

    elo: float = 1500.0
    goals_for: float = 1.4
    goals_against: float = 1.4
    sot_for: float = 4.5
    sot_against: float = 4.5


ELO_K = 20.0
ELO_HOME = 60.0
FORM_WEIGHT = 0.1
"""Weight of the newest game in each average: about the last 10 games count."""


def team_form(games: pl.DataFrame) -> pl.DataFrame:
    """The games with FORM_COLUMNS: home minus away, each from games before this one only.

    Teams carry their form across seasons within a league. Shots on target missing from
    a file count as the team's own average, so they leave its form unchanged.
    """
    form: defaultdict[tuple[str, str], _Form] = defaultdict(_Form)
    rows = []
    for g in games.iter_rows(named=True):
        home, away = form[(g["league"], g["fd_home"])], form[(g["league"], g["fd_away"])]
        rows.append(
            (
                home.elo - away.elo,
                home.goals_for - away.goals_for,
                home.goals_against - away.goals_against,
                home.sot_for - away.sot_for,
                home.sot_against - away.sot_against,
            )
        )
        _update(home, away, g)
    return games.hstack(pl.DataFrame(rows, schema=FORM_COLUMNS, orient="row"))


def _update(home: _Form, away: _Form, game: dict) -> None:
    hg, ag = game["home_goals"], game["away_goals"]
    expected = 1 / (1 + 10 ** (-(home.elo + ELO_HOME - away.elo) / 400))
    actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    # Bigger wins move ratings more, as in World Football Elo.
    shift = ELO_K * np.log2(abs(hg - ag) + 1 if hg != ag else 2) * (actual - expected)
    home.elo, away.elo = home.elo + shift, away.elo - shift

    home_sot = game["home_shots_on_target"]
    away_sot = game["away_shots_on_target"]
    home_sot = home.sot_for if home_sot is None else home_sot
    away_sot = away.sot_for if away_sot is None else away_sot

    def blend(old: float, new: float) -> float:
        return (1 - FORM_WEIGHT) * old + FORM_WEIGHT * new

    home.goals_for, home.goals_against = blend(home.goals_for, hg), blend(home.goals_against, ag)
    away.goals_for, away.goals_against = blend(away.goals_for, ag), blend(away.goals_against, hg)
    home.sot_for, home.sot_against = (
        blend(home.sot_for, home_sot),
        blend(home.sot_against, away_sot),
    )
    away.sot_for, away.sot_against = (
        blend(away.sot_for, away_sot),
        blend(away.sot_against, home_sot),
    )


# --- Scoring every forecaster walk-forward --------------------------------------------


def _market_features(probabilities: np.ndarray) -> np.ndarray:
    """log(P(H)/P(A)) and log(P(D)/P(A)): the market's view as two numbers."""
    return np.log(probabilities[:, :2] / probabilities[:, 2:])


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Multinomial logistic regression; probabilities in OUTCOMES order."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train_x, train_y)
    order = [list(model.classes_).index(k) for k in range(len(OUTCOMES))]
    return model.predict_proba(test_x)[:, order]


def forecasts(games: pl.DataFrame, first_test_season: int = FIRST_TEST_SEASON) -> dict:
    """Each forecaster's probabilities for the test seasons, and those games' rows.

    Returns {"games": test rows, name: array (games x OUTCOMES)} for every FORECASTERS.
    """
    games = team_form(games)
    y = games["result"].replace_strict({k: i for i, k in enumerate(OUTCOMES)}).to_numpy()
    early = margin_free(games.select(f"early_{k}" for k in OUTCOMES).to_numpy())
    close = margin_free(games.select(f"close_{k}" for k in OUTCOMES).to_numpy())
    market = _market_features(early)
    form = games.select(FORM_COLUMNS).to_numpy()
    inputs = {
        "market_refit": market,
        "market_and_form": np.column_stack([market, form]),
        "form_only": form,
    }

    seasons = games["season_start"].to_numpy()
    test = seasons >= first_test_season
    out: dict = {
        "games": games.filter(pl.Series(test)),
        "early": early[test],
        "close": close[test],
    }
    for name, x in inputs.items():
        parts = []
        for season in sorted(set(seasons[test].tolist())):
            train, now = seasons < season, seasons == season
            parts.append(_fit_predict(x[train], y[train], x[now]))
        out[name] = np.concatenate(parts)
    return out


def losses(forecast: dict) -> pl.DataFrame:
    """One row per test game and forecaster: league, season_start, log_loss, rps."""
    games = forecast["games"]
    y = games["result"].replace_strict({k: i for i, k in enumerate(OUTCOMES)}).to_numpy()
    rows = np.arange(len(y))
    actual = np.eye(len(OUTCOMES))[y]
    frames = []
    for name in FORECASTERS:
        p = forecast[name]
        # RPS treats H, D, A as ordered: calling a home win a draw costs less than an away win.
        rps = (((p.cumsum(axis=1) - actual.cumsum(axis=1))[:, :-1]) ** 2).sum(axis=1) / 2
        frames.append(
            games.select("league", "season_start").with_columns(
                forecaster=pl.lit(name),
                game=pl.Series(rows),
                log_loss=pl.Series(-np.log(p[rows, y])),
                rps=pl.Series(rps),
            )
        )
    return pl.concat(frames)


def paired_differences(
    table: pl.DataFrame, pairs: list[tuple[str, str]], by: list[str] | None = None
) -> pl.DataFrame:
    """Mean of (a - b) per game for each pair, with a 95% interval, overall or per `by`.

    Negative means a forecasts better than b. Columns: a, b, *by, games, then for log
    loss and RPS the mean difference and its interval (low, high).
    """
    keys = ["game", "league", "season_start"]
    frames = []
    for a, b in pairs:
        joined = table.filter(pl.col("forecaster") == a).join(
            table.filter(pl.col("forecaster") == b), on=keys, suffix="_b"
        )
        diff = joined.select(
            *keys,
            (pl.col("log_loss") - pl.col("log_loss_b")).alias("log_loss"),
            (pl.col("rps") - pl.col("rps_b")).alias("rps"),
        )
        aggs = [pl.len().alias("games")]
        for metric in ("log_loss", "rps"):
            half = 1.96 * pl.col(metric).std() / pl.len().sqrt()
            aggs += [
                pl.col(metric).mean().alias(metric),
                (pl.col(metric).mean() - half).alias(f"{metric}_low"),
                (pl.col(metric).mean() + half).alias(f"{metric}_high"),
            ]
        grouped = (
            diff.group_by(by, maintain_order=True).agg(aggs).sort(by) if by else diff.select(aggs)
        )
        frames.append(grouped.select(pl.lit(a).alias("a"), pl.lit(b).alias("b"), pl.all()))
    return pl.concat(frames)


COMPARISONS = [
    ("close", "early"),
    ("market_refit", "early"),
    ("market_and_form", "early"),
    ("market_and_form", "close"),
    ("form_only", "close"),
]
"""The questions: is the close sharper than the early price? Can a model improve on
the early price (where a bettor acts), or on the close (the hard bar)?"""


def margin_methods(games: pl.DataFrame) -> pl.DataFrame:
    """method, timing, log_loss: how well each way of removing the margin forecasts."""
    test = games.filter(pl.col("season_start") >= FIRST_TEST_SEASON)
    y = test["result"].replace_strict({k: i for i, k in enumerate(OUTCOMES)}).to_numpy()
    rows = []
    for timing in ("early", "close"):
        odds = test.select(f"{timing}_{k}" for k in OUTCOMES).to_numpy()
        for method in ("proportional", "power", "shin"):
            p = margin_free(odds, method)
            rows.append((method, timing, float(-np.log(p[np.arange(len(y)), y]).mean())))
    return pl.DataFrame(rows, schema=["method", "timing", "log_loss"], orient="row")


def main(settings: Settings = default_settings) -> pl.DataFrame:
    """Score every forecaster and print the comparisons, overall and per league."""
    games = read_games(settings.modeling.odds_dir)
    table = losses(forecasts(games))
    with pl.Config(float_precision=5, tbl_rows=60, tbl_cols=12):
        print(margin_methods(games))
        print(table.group_by("forecaster").agg(pl.col("log_loss", "rps").mean()).sort("log_loss"))
        print(paired_differences(table, COMPARISONS))
        print(paired_differences(table, [("market_and_form", "early")], by=["league"]))
    return table


def exec_market_test() -> None:
    """Entry point `market_test`: can a model beat the market's probabilities?"""
    main()


__all__ = [
    "COMPARISONS",
    "margin_methods",
    "FORECASTERS",
    "forecasts",
    "losses",
    "margin_free",
    "paired_differences",
    "read_games",
    "team_form",
]
