"""Double Poisson match model driven by pre-match Elo, with optional Dixon-Coles.

Home and away goals are modelled as two Poisson variables whose log-rates are
linear in the pre-match Elo difference. Outcome probabilities (H/D/A) come from
the joint score grid. Everything is walk-forward: each season is predicted by a
model fitted only on earlier seasons, and Elo ratings are always pre-match.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from pandera.typing.polars import DataFrame
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from money_printer_research.cool_stuff import returns
from money_printer_research.schema import (
    CLEANED_MATCH,
    OUTCOMES,
    EloCols,
    EloInputCols,
    PredictionCols,
)

E = EloCols
P = PredictionCols

HOME_GOALS = E.full_time_home_goals
AWAY_GOALS = E.full_time_away_goals
RESULT = E.full_time_result
ELO_SCALE = 400.0
"""Elo points per factor of 10 in the odds; also scales elo_diff into the regressors."""

PROMOTED_GAP = 80.0
"""Points below base a promoted team starts at when no team went down (the first season)."""

MAX_GOALS = 10
"""Largest score per side in the grid; observed scores above it are capped to it."""

RHO_GRID = np.linspace(-0.25, 0.1, 36)
"""Dixon-Coles rho values _fit_rho tries."""

EPS = 1e-12
"""Floor on probabilities before taking a log."""

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class EloParams:
    """Elo settings.

    Tuned on 2005/06-2014/15 by walk-forward log loss. The surface is flat:
    K 10-20, home_adv 20-80 and regress 0-0.2 all land within 0.001.
    """

    k: float = 15.0
    """Update size."""
    home_adv: float = 40.0
    """Elo points added to the home side in the expectation."""
    season_regress: float = 0.0
    """Share pulled back toward base at each season start."""
    base: float = 1500.0
    """Mean rating."""


DEFAULT_ELO = EloParams()


def _goal_multiplier(gd: int) -> float:
    """Scale an Elo update by the goal difference: 1, 1.5, then (11 + gd) / 8."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def _teams_by_season(matches: DataFrame[EloInputCols]) -> dict[str, set[str]]:
    """Season -> every team that plays in it, home or away."""
    teams = pl.concat(
        [
            matches.select(E.season, pl.col(side).alias("team"))
            for side in (E.home_team, E.away_team)
        ]
    )
    return {season: set(group["team"].to_list()) for (season,), group in teams.group_by(E.season)}


@returns(EloInputCols)
def to_elo_input(matches: DataFrame[CLEANED_MATCH]) -> pl.DataFrame:
    """stat_matches -> EloInputCols: season label, date only, full-time goals and result."""
    EI = EloInputCols
    start = pl.col(CLEANED_MATCH.season_start)
    hg, ag = pl.col(CLEANED_MATCH.home_goals), pl.col(CLEANED_MATCH.away_goals)
    return matches.select(
        pl.format("{}/{}", start, (start + 1).cast(pl.String).str.slice(2, 2)).alias(EI.season),
        pl.col(CLEANED_MATCH.date).dt.date().alias(EI.match_date),
        pl.col(CLEANED_MATCH.home_team).alias(EI.home_team),
        pl.col(CLEANED_MATCH.away_team).alias(EI.away_team),
        hg.alias(EI.full_time_home_goals),
        ag.alias(EI.full_time_away_goals),
        pl.when(hg > ag)
        .then(pl.lit("H"))
        .when(hg == ag)
        .then(pl.lit("D"))
        .otherwise(pl.lit("A"))
        .alias(EI.full_time_result),
    )


def get_elo(
    matches: DataFrame[EloInputCols], params: EloParams = DEFAULT_ELO
) -> DataFrame[EloCols]:
    """Add pre-match elo_home, elo_away and elo_diff (home minus away, no home bonus).

    Promoted teams start each season at the mean end-of-season rating of the
    teams that went down, which keeps the league average stable.
    """
    df = DataFrame[EloInputCols](matches.sort(E.match_date))
    teams_by_season = _teams_by_season(df)
    rating: dict[str, float] = {}
    prev_teams: set[str] = set()
    current_season: str | None = None
    elo_h: list[float] = []
    elo_a: list[float] = []

    rows = df.select(E.season, E.home_team, E.away_team, HOME_GOALS, AWAY_GOALS).iter_rows()
    for season, home, away, hg, ag in rows:
        if season != current_season:
            this_season = teams_by_season[season]
            relegated = prev_teams - this_season
            promoted_start = (
                float(np.mean([rating[t] for t in relegated]))
                if relegated
                else params.base - PROMOTED_GAP
            )
            for t in this_season - prev_teams:
                rating[t] = promoted_start
            for t in this_season:
                rating[t] = params.base + (rating[t] - params.base) * (1 - params.season_regress)
            prev_teams, current_season = this_season, season

        rh, ra = rating[home], rating[away]
        elo_h.append(rh)
        elo_a.append(ra)

        expected = 1 / (1 + 10 ** (-(rh + params.home_adv - ra) / ELO_SCALE))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = params.k * _goal_multiplier(hg - ag) * (score - expected)
        rating[home] = rh + delta
        rating[away] = ra - delta

    out = df.with_columns(pl.Series(E.elo_home, elo_h), pl.Series(E.elo_away, elo_a)).with_columns(
        (pl.col(E.elo_home) - pl.col(E.elo_away)).alias(E.elo_diff)
    )
    return EloCols.validate(out, lazy=True)


def _dc_tau(lam: FloatArray, mu: FloatArray, rho: float, max_goals: int) -> FloatArray:
    """Dixon-Coles adjustment for the four low scores; shape (n, g, g)."""
    tau = np.ones((len(lam), max_goals + 1, max_goals + 1))
    tau[:, 0, 0] = 1 - lam * mu * rho
    tau[:, 0, 1] = 1 + lam * rho
    tau[:, 1, 0] = 1 + mu * rho
    tau[:, 1, 1] = 1 - rho
    return tau


def score_grid(
    lam: FloatArray, mu: FloatArray, rho: float = 0.0, max_goals: int = MAX_GOALS
) -> FloatArray:
    """Joint probability of each scoreline; grid[i, h, a] = P(home=h, away=a)."""
    goals = np.arange(max_goals + 1)
    # scipy's stubs type pmf as float; with array inputs it returns an (n, g) array.
    ph = _as_float(poisson.pmf(goals[None, :], lam[:, None]))
    pa = _as_float(poisson.pmf(goals[None, :], mu[:, None]))
    grid = ph[:, :, None] * pa[:, None, :]
    if rho:
        grid = grid * _dc_tau(lam, mu, rho, max_goals)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcome_probs(grid: FloatArray) -> FloatArray:
    """(n, 3) probabilities in OUTCOMES order: home win, draw, away win."""
    shape = grid.shape[1:]
    home = np.tril(np.ones(shape), k=-1)  # h > a
    draw = np.eye(shape[0])
    away = np.triu(np.ones(shape), k=1)  # a > h
    return np.stack([(grid * m).sum(axis=(1, 2)) for m in (home, draw, away)], axis=1)


def _fit_rho(lam: FloatArray, mu: FloatArray, hg: IntArray, ag: IntArray) -> float:
    """Pick rho by maximum likelihood of the observed scores on RHO_GRID."""
    rows = np.arange(len(lam))
    capped_h, capped_a = np.minimum(hg, MAX_GOALS), np.minimum(ag, MAX_GOALS)
    best, best_ll = 0.0, -np.inf
    for rho in RHO_GRID:
        p = score_grid(lam, mu, float(rho))[rows, capped_h, capped_a]
        ll = np.log(np.clip(p, EPS, None)).sum()
        if ll > best_ll:
            best, best_ll = float(rho), ll
    return best


def _as_float(x: object) -> FloatArray:
    """Coerce to a float64 array (sklearn and scipy stubs return loose types)."""
    return np.asarray(x, dtype=np.float64)


def _features(df: pl.DataFrame) -> FloatArray:
    """Regressor matrix: elo_diff in units of ELO_SCALE, shape (n, 1)."""
    return (df[E.elo_diff].to_numpy() / ELO_SCALE).reshape(-1, 1)


def _goals(df: pl.DataFrame, col: str) -> IntArray:
    """Goals column as int64."""
    return df[col].to_numpy().astype(np.int64)


def walk_forward(
    df: DataFrame[EloCols], test_seasons: list[str], dixon_coles: bool = True
) -> DataFrame[PredictionCols]:
    """Predict each test season with a model fitted on all earlier seasons."""
    seasons = sorted(df[E.season].unique().to_list())
    out: list[pl.DataFrame] = []
    for season in test_seasons:
        train = df.filter(pl.col(E.season).is_in([s for s in seasons if s < season]))
        test = df.filter(pl.col(E.season) == season)
        x_tr, x_te = _features(train), _features(test)
        hg_tr, ag_tr = _goals(train, HOME_GOALS), _goals(train, AWAY_GOALS)

        home_m = PoissonRegressor(alpha=0).fit(x_tr, hg_tr)
        away_m = PoissonRegressor(alpha=0).fit(x_tr, ag_tr)

        rho = 0.0
        if dixon_coles:
            rho = _fit_rho(
                _as_float(home_m.predict(x_tr)), _as_float(away_m.predict(x_tr)), hg_tr, ag_tr
            )

        lam = _as_float(home_m.predict(x_te))
        mu = _as_float(away_m.predict(x_te))
        probs = outcome_probs(score_grid(lam, mu, rho))
        out.append(
            test.with_columns(
                pl.Series(P.lam_home, lam),
                pl.Series(P.lam_away, mu),
                pl.Series(P.p_home, probs[:, 0]),
                pl.Series(P.p_draw, probs[:, 1]),
                pl.Series(P.p_away, probs[:, 2]),
                pl.lit(rho).alias(P.rho),
            )
        )
    return PredictionCols.validate(pl.concat(out), lazy=True)


def evaluate(pred: DataFrame[PredictionCols]) -> dict[str, float]:
    """Log loss, ranked probability score and accuracy of p_home/p_draw/p_away."""
    p = pred.select(P.p_home, P.p_draw, P.p_away).to_numpy()
    y = np.array([OUTCOMES.index(r) for r in pred[RESULT].to_list()])
    onehot = np.eye(len(OUTCOMES))[y]
    log_loss = float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, None)).mean())
    # RPS: squared gaps of the cumulative distributions; the last one is always 0.
    rps = float((((np.cumsum(p, 1) - np.cumsum(onehot, 1))[:, :-1]) ** 2).sum(1).mean() / 2)
    acc = float((p.argmax(1) == y).mean())
    return {"log_loss": log_loss, "rps": rps, "accuracy": acc, "n": len(y)}
