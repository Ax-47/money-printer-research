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
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

HOME_GOALS = "full_time_home_goals"
AWAY_GOALS = "full_time_away_goals"
RESULT = "full_time_result"
OUTCOMES = ["H", "D", "A"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EloParams:
    # Tuned on 2005/06-2014/15 by walk-forward log loss. The surface is flat:
    # K 10-20, home_adv 20-80 and regress 0-0.2 all land within 0.001.
    k: float = 15.0  # update size
    home_adv: float = 40.0  # Elo points added to the home side in the expectation
    season_regress: float = 0.0  # share pulled back toward 1500 at each season start
    base: float = 1500.0


def _goal_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


DEFAULT_ELO = EloParams()


def add_elo(matches: pl.DataFrame, params: EloParams = DEFAULT_ELO) -> pl.DataFrame:
    """Add pre-match elo_home, elo_away and elo_diff (home minus away, no home bonus).

    Promoted teams start each season at the mean end-of-season rating of the
    teams that went down, which keeps the league average stable.
    """
    df = matches.sort("match_date")
    rating: dict[str, float] = {}
    prev_teams: set[str] = set()
    current_season = None
    elo_h, elo_a = [], []

    for season, home, away, hg, ag in df.select(
        "season", "home_team", "away_team", HOME_GOALS, AWAY_GOALS
    ).iter_rows():
        if season != current_season:
            this_season = set(
                df.filter(pl.col("season") == season)
                .select(pl.concat_list("home_team", "away_team").explode())
                .to_series()
                .to_list()
            )
            relegated = prev_teams - this_season
            promoted_start = (
                float(np.mean([rating[t] for t in relegated])) if relegated else params.base - 80
            )
            for t in this_season - prev_teams:
                rating[t] = promoted_start
            for t in this_season:
                rating[t] = params.base + (rating[t] - params.base) * (1 - params.season_regress)
            prev_teams, current_season = this_season, season

        rh, ra = rating[home], rating[away]
        elo_h.append(rh)
        elo_a.append(ra)

        expected = 1 / (1 + 10 ** (-(rh + params.home_adv - ra) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = params.k * _goal_multiplier(hg - ag) * (score - expected)
        rating[home] = rh + delta
        rating[away] = ra - delta

    return df.with_columns(
        pl.Series("elo_home", elo_h), pl.Series("elo_away", elo_a)
    ).with_columns((pl.col("elo_home") - pl.col("elo_away")).alias("elo_diff"))


def _dc_tau(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int) -> np.ndarray:
    """Dixon-Coles adjustment for the four low scores; shape (n, g, g)."""
    tau = np.ones((len(lam), max_goals + 1, max_goals + 1))
    tau[:, 0, 0] = 1 - lam * mu * rho
    tau[:, 0, 1] = 1 + lam * rho
    tau[:, 1, 0] = 1 + mu * rho
    tau[:, 1, 1] = 1 - rho
    return tau


def score_grid(
    lam: np.ndarray, mu: np.ndarray, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Joint probability of each scoreline; grid[i, h, a] = P(home=h, away=a)."""
    goals = np.arange(max_goals + 1)
    # scipy's stubs type pmf as float; with array inputs it returns an (n, g) array.
    ph = np.asarray(poisson.pmf(goals[None, :], lam[:, None]), dtype=float)
    pa = np.asarray(poisson.pmf(goals[None, :], mu[:, None]), dtype=float)
    grid = ph[:, :, None] * pa[:, None, :]
    if rho:
        grid = grid * _dc_tau(lam, mu, rho, max_goals)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcome_probs(grid: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in OUTCOMES order: home win, draw, away win."""
    home = np.tril(np.ones(grid.shape[1:]), k=-1)  # h > a
    draw = np.eye(grid.shape[1])
    away = np.triu(np.ones(grid.shape[1:]), k=1)  # a > h
    return np.stack([(grid * m).sum(axis=(1, 2)) for m in (home, draw, away)], axis=1)


def _fit_rho(lam: np.ndarray, mu: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Pick rho by maximum likelihood of the observed scores on a small grid."""
    best, best_ll = 0.0, -np.inf
    capped_h, capped_a = np.minimum(hg, 10), np.minimum(ag, 10)
    for rho in np.linspace(-0.25, 0.1, 36):
        grid = score_grid(lam, mu, rho)
        p = grid[np.arange(len(lam)), capped_h, capped_a]
        ll = np.log(np.clip(p, 1e-12, None)).sum()
        if ll > best_ll:
            best, best_ll = float(rho), ll
    return best


def _as_float(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def walk_forward(
    df: pl.DataFrame, test_seasons: list[str], dixon_coles: bool = True
) -> pl.DataFrame:
    """Predict each test season with a model fitted on all earlier seasons."""
    seasons = sorted(df["season"].unique().to_list())
    out = []
    for season in test_seasons:
        train = df.filter(pl.col("season").is_in([s for s in seasons if s < season]))
        test = df.filter(pl.col("season") == season)
        x_tr = (train["elo_diff"].to_numpy() / 400).reshape(-1, 1)
        x_te = (test["elo_diff"].to_numpy() / 400).reshape(-1, 1)
        hg_tr = train[HOME_GOALS].to_numpy().astype(np.int64)
        ag_tr = train[AWAY_GOALS].to_numpy().astype(np.int64)

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
                pl.Series("lam_home", lam),
                pl.Series("lam_away", mu),
                pl.Series("p_home", probs[:, 0]),
                pl.Series("p_draw", probs[:, 1]),
                pl.Series("p_away", probs[:, 2]),
                pl.lit(rho).alias("rho"),
            )
        )
    return pl.concat(out)


def evaluate(pred: pl.DataFrame) -> dict[str, float]:
    """Log loss, ranked probability score and accuracy of p_home/p_draw/p_away."""
    p = pred.select("p_home", "p_draw", "p_away").to_numpy()
    y = np.array([OUTCOMES.index(r) for r in pred[RESULT].to_list()])
    onehot = np.eye(3)[y]
    log_loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
    rps = float((((np.cumsum(p, 1) - np.cumsum(onehot, 1))[:, :2]) ** 2).sum(1).mean() / 2)
    acc = float((p.argmax(1) == y).mean())
    return {"log_loss": log_loss, "rps": rps, "accuracy": acc, "n": len(y)}
