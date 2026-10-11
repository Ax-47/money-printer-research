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
    CleanedMatchCols,
    OUTCOMES,
    EloCols,
    EloInputCols,
    PredictionCols,
)


HOME_GOALS = EloCols.full_time_home_goals
AWAY_GOALS = EloCols.full_time_away_goals
RESULT = EloCols.full_time_result
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
            matches.select(EloCols.season, pl.col(side).alias("team"))
            for side in (EloCols.home_team, EloCols.away_team)
        ]
    )
    return {
        season: set(group["team"].to_list()) for (season,), group in teams.group_by(EloCols.season)
    }


def season_label(season_start: pl.Expr) -> pl.Expr:
    """EloInputCols.season from a season's start year: 2014 -> "2014/15"."""
    return pl.format("{}/{}", season_start, (season_start + 1).cast(pl.String).str.slice(2, 2))


@returns(EloInputCols)
def to_elo_input(matches: DataFrame[CleanedMatchCols]) -> pl.DataFrame:
    """cleaned_matches -> EloInputCols: season label, date only, full-time goals and result."""
    ei = EloInputCols
    start = pl.col(CleanedMatchCols.season_start)
    hg, ag = pl.col(CleanedMatchCols.home_goals), pl.col(CleanedMatchCols.away_goals)
    return matches.select(
        season_label(start).alias(ei.season),
        pl.col(CleanedMatchCols.date).dt.date().alias(ei.match_date),
        pl.col(CleanedMatchCols.home_team).alias(ei.home_team),
        pl.col(CleanedMatchCols.away_team).alias(ei.away_team),
        hg.alias(ei.full_time_home_goals),
        ag.alias(ei.full_time_away_goals),
        pl.when(hg > ag)
        .then(pl.lit("H"))
        .when(hg == ag)
        .then(pl.lit("D"))
        .otherwise(pl.lit("A"))
        .alias(ei.full_time_result),
    )


def _expected_home_score(home_rating: float, away_rating: float, home_adv: float) -> float:
    """Home side's expected score (win 1, draw 0.5) from the ratings and home advantage."""
    return 1 / (1 + 10 ** (-(home_rating + home_adv - away_rating) / ELO_SCALE))


def _actual_home_score(home_goals: int, away_goals: int) -> float:
    """1 for a home win, 0.5 for a draw, 0 for an away win."""
    return 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0


def _start_season(
    rating: dict[str, float],
    last_season_teams: set[str],
    this_season_teams: set[str],
    params: EloParams,
) -> None:
    """Rate the promoted teams and pull every rating toward base, in place.

    Promoted teams start at the mean rating of the teams that went down, or
    PROMOTED_GAP below base when none did (the first season).
    """
    relegated = last_season_teams - this_season_teams
    promoted_start = (
        float(np.mean([rating[t] for t in relegated])) if relegated else params.base - PROMOTED_GAP
    )
    for team in this_season_teams - last_season_teams:
        rating[team] = promoted_start
    for team in this_season_teams:
        rating[team] = params.base + (rating[team] - params.base) * (1 - params.season_regress)


def _replay(
    df: DataFrame[EloInputCols], params: EloParams
) -> tuple[list[float], list[float], dict[str, float], set[str], str | None]:
    """Walk the played matches in date order.

    Returns the pre-match home and away ratings of each match, then the state after the
    last one: every team's rating, the teams of the last season and that season.
    """
    teams_by_season = _teams_by_season(df)
    rating: dict[str, float] = {}
    last_season_teams: set[str] = set()
    current_season: str | None = None
    elo_home: list[float] = []
    elo_away: list[float] = []

    rows = df.select(
        EloCols.season, EloCols.home_team, EloCols.away_team, HOME_GOALS, AWAY_GOALS
    ).iter_rows()
    for season, home, away, home_goals, away_goals in rows:
        if season != current_season:
            this_season_teams = teams_by_season[season]
            _start_season(rating, last_season_teams, this_season_teams, params)
            last_season_teams, current_season = this_season_teams, season

        home_rating, away_rating = rating[home], rating[away]
        elo_home.append(home_rating)
        elo_away.append(away_rating)

        expected = _expected_home_score(home_rating, away_rating, params.home_adv)
        actual = _actual_home_score(home_goals, away_goals)
        margin = _goal_multiplier(home_goals - away_goals)
        delta = params.k * margin * (actual - expected)
        rating[home] = home_rating + delta
        rating[away] = away_rating - delta
    return elo_home, elo_away, rating, last_season_teams, current_season


def _with_diff(df: pl.DataFrame, elo_home: list[float], elo_away: list[float]) -> pl.DataFrame:
    return df.with_columns(
        pl.Series(EloCols.elo_home, elo_home, pl.Float64),
        pl.Series(EloCols.elo_away, elo_away, pl.Float64),
    ).with_columns((pl.col(EloCols.elo_home) - pl.col(EloCols.elo_away)).alias(EloCols.elo_diff))


def get_elo(
    matches: DataFrame[EloInputCols], params: EloParams = DEFAULT_ELO
) -> DataFrame[EloCols]:
    """Add pre-match elo_home, elo_away and elo_diff (home minus away, no home bonus).

    Promoted teams start each season at the mean end-of-season rating of the
    teams that went down, which keeps the league average stable.
    """
    df = DataFrame[EloInputCols](matches.sort(EloCols.match_date))
    elo_home, elo_away, *_ = _replay(df, params)
    return EloCols.validate(_with_diff(df, elo_home, elo_away), lazy=True)


def fixture_elo(
    played: DataFrame[EloInputCols], fixtures: pl.DataFrame, params: EloParams = DEFAULT_ELO
) -> pl.DataFrame:
    """Pre-match elo_home, elo_away and elo_diff for unplayed fixtures.

    `fixtures` has EloInputCols' season, match_date, home_team and away_team. The ratings
    are those after every played match; a fixture has no result, so it moves none. A
    fixture in a season the played matches have not reached starts that season first
    (promoted teams, regression toward base), as get_elo would.
    """
    played = DataFrame[EloInputCols](played.sort(EloCols.match_date))
    *_, rating, last_season_teams, current_season = _replay(played, params)
    cols = [EloCols.season, EloCols.home_team, EloCols.away_team]
    teams_by_season = _teams_by_season(pl.concat([played.select(cols), fixtures.select(cols)]))  # pyright: ignore[reportArgumentType]
    fixtures = fixtures.sort(EloCols.match_date)
    elo_home: list[float] = []
    elo_away: list[float] = []
    for season, home, away in fixtures.select(cols).iter_rows():
        if season != current_season:
            this_season_teams = teams_by_season[season]
            _start_season(rating, last_season_teams, this_season_teams, params)
            last_season_teams, current_season = this_season_teams, season
        elo_home.append(rating[home])
        elo_away.append(rating[away])
    return _with_diff(fixtures, elo_home, elo_away)


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
    return (df[EloCols.elo_diff].to_numpy() / ELO_SCALE).reshape(-1, 1)


def _goals(df: pl.DataFrame, col: str) -> IntArray:
    """Goals column as int64."""
    return df[col].to_numpy().astype(np.int64)


def _predict_season(train: pl.DataFrame, test: pl.DataFrame, dixon_coles: bool) -> pl.DataFrame:
    """Fit home and away goal models on `train`; add rates and H/D/A probabilities to `test`."""
    x_train, x_test = _features(train), _features(test)
    home_goals, away_goals = _goals(train, HOME_GOALS), _goals(train, AWAY_GOALS)
    home_model = PoissonRegressor(alpha=0).fit(x_train, home_goals)
    away_model = PoissonRegressor(alpha=0).fit(x_train, away_goals)

    rho = 0.0
    if dixon_coles:
        rho = _fit_rho(
            _as_float(home_model.predict(x_train)),
            _as_float(away_model.predict(x_train)),
            home_goals,
            away_goals,
        )

    lam = _as_float(home_model.predict(x_test))
    mu = _as_float(away_model.predict(x_test))
    probs = outcome_probs(score_grid(lam, mu, rho))
    return test.with_columns(
        pl.Series(PredictionCols.lam_home, lam),
        pl.Series(PredictionCols.lam_away, mu),
        pl.Series(PredictionCols.p_home, probs[:, 0]),
        pl.Series(PredictionCols.p_draw, probs[:, 1]),
        pl.Series(PredictionCols.p_away, probs[:, 2]),
        pl.lit(rho).alias(PredictionCols.rho),
    )


def walk_forward(
    df: DataFrame[EloCols], test_seasons: list[str], dixon_coles: bool = True
) -> DataFrame[PredictionCols]:
    """Predict each test season with a model fitted on all earlier seasons."""
    seasons = sorted(df[EloCols.season].unique().to_list())
    predictions: list[pl.DataFrame] = []
    for season in test_seasons:
        earlier = [s for s in seasons if s < season]
        train = df.filter(pl.col(EloCols.season).is_in(earlier))
        test = df.filter(pl.col(EloCols.season) == season)
        predictions.append(_predict_season(train, test, dixon_coles))
    return PredictionCols.validate(pl.concat(predictions), lazy=True)


def evaluate(pred: DataFrame[PredictionCols]) -> dict[str, float]:
    """Log loss, ranked probability score and accuracy of p_home/p_draw/p_away."""
    p = pred.select(PredictionCols.p_home, PredictionCols.p_draw, PredictionCols.p_away).to_numpy()
    y = np.array([OUTCOMES.index(r) for r in pred[RESULT].to_list()])
    onehot = np.eye(len(OUTCOMES))[y]
    log_loss = float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, None)).mean())
    # RPS: squared gaps of the cumulative distributions; the last one is always 0.
    rps = float((((np.cumsum(p, 1) - np.cumsum(onehot, 1))[:, :-1]) ** 2).sum(1).mean() / 2)
    acc = float((p.argmax(1) == y).mean())
    return {"log_loss": log_loss, "rps": rps, "accuracy": acc, "n": len(y)}
