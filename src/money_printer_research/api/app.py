"""FastAPI app serving what clean_data wrote to settings.clean.out_dir.

    GET /health                      liveness, and whether the clean tables exist
    GET /tables                      the CSV tables in the clean directory
    GET /tables/{name}               one table's rows, paged
    GET /predictions/seasons         seasons the Elo model can predict
    GET /predictions?season=...      walk_forward H/D/A probabilities for one season
    GET /predictions/metrics?...     log loss, RPS and accuracy for that season
    GET /upcoming?league=...&team=...  the `predict` command's H/D/A for unplayed fixtures
    GET /backtest?min_ev=...&league=...  the `backtest` command's results for one EV threshold,
                                     and for thresholds chosen season by season
    GET /analysis?league=...         the `report` command's results: model vs market,
                                     strategies (one league, or ALL), staking, practicality
    GET /live?league=...             today's games from ESPN: state, score, and H/D/A
                                     before and after the starting XIs were announced
    GET /live/{espn_id}              one game, with both XIs and each starter's form
    GET /teams?league=...            every team, or one league's current table
    GET /teams/{team}                a team's season: record, home/away, form, Elo
    GET /, /backtest.html, /analysis.html  web pages over /upcoming, /backtest, /analysis

Tables, upcoming.csv, backtest.csv and analysis.json are read from disk on each request,
so a new clean_data, predict, backtest or report run shows up without a restart.
Predictions are cached per season until elo.csv changes.

With settings.live.enabled the app polls ESPN every poll_seconds while it runs
(live.poller.refresh) and writes settings.live.db_file, which /live reads.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pandera.typing.polars import DataFrame

from money_printer_research.analysis.backtest import (
    cumulative_profit,
    log_loss_vs_market,
    summary,
    walk_forward_thresholds,
)
from money_printer_research.api import teams as team_tables
from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.features.elo import evaluate, walk_forward
from money_printer_research.live.poller import Predictor, refresh
from money_printer_research.live.store import Store
from money_printer_research.schema import EloCols, PredictionCols

ELO_TABLE = "elo"
MATCHES_TABLE = "cleaned_matches"
log = logging.getLogger(__name__)
MAX_LIMIT = 1000
STATIC = Path(__file__).parent / "static"


def _read_output(path: Path, command: str) -> pl.DataFrame:
    """A CSV a pipeline command writes, or a 404 saying which command to run."""
    if not path.is_file():
        raise HTTPException(404, f"no {path}; run `uv run {command}` first")
    return pl.read_csv(path, try_parse_dates=True)


def _rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts; NaN becomes null, since JSON has no NaN."""
    return df.fill_nan(None).to_dicts()


def _involving(df: pl.DataFrame, team: str) -> pl.DataFrame:
    """The rows where `team` plays, home or away."""
    return df.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))


def _predictable(elo: DataFrame[EloCols]) -> list[str]:
    """Every season but the first: walk_forward needs an earlier season to fit on."""
    return sorted(elo[EloCols.season].unique().to_list())[1:]


class _Tables:
    """The clean CSVs, and the Elo predictions cached on elo.csv's mtime."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._elo_mtime: float | None = None
        self._predictions: dict[tuple[str, bool], DataFrame[PredictionCols]] = {}

    def names(self) -> list[str]:
        if not self.out_dir.is_dir():
            return []
        return sorted(p.stem for p in self.out_dir.glob("*.csv"))

    def path(self, name: str) -> Path:
        # Only names listed in the directory, so a request cannot reach other files.
        if name not in self.names():
            raise HTTPException(404, f"no table {name!r}; see GET /tables")
        return self.out_dir / f"{name}.csv"

    def read(self, name: str) -> pl.DataFrame:
        return pl.read_csv(self.path(name), infer_schema_length=None, try_parse_dates=True)

    def elo(self) -> DataFrame[EloCols]:
        return DataFrame[EloCols](EloCols.validate(self.read(ELO_TABLE)))

    def seasons(self) -> list[str]:
        return _predictable(self.elo())

    def predictions(self, season: str, dixon_coles: bool) -> DataFrame[PredictionCols]:
        mtime = self.path(ELO_TABLE).stat().st_mtime
        if mtime != self._elo_mtime:
            self._elo_mtime, self._predictions = mtime, {}
        key = (season, dixon_coles)
        if key not in self._predictions:
            elo = self.elo()
            if season not in _predictable(elo):
                raise HTTPException(
                    404, f"no predictions for season {season!r}; see GET /predictions/seasons"
                )
            self._predictions[key] = walk_forward(elo, [season], dixon_coles)
        return self._predictions[key]


class _Live:
    """The live database, opened on first use so an app that never touches it makes no file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store: Store | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.settings.live.db_file)
        return self._store

    async def poll_forever(self) -> None:
        predictor = Predictor(self.settings)
        while True:
            try:
                seen = await asyncio.to_thread(refresh, self.settings, self.store, predictor)
                log.info("live poll: %d games today", seen)
            except Exception:
                log.exception("live poll failed")
            await asyncio.sleep(self.settings.live.poll_seconds)


def create_app(settings: Settings = default_settings) -> FastAPI:
    """The API over settings.clean.out_dir, and the live database when it is enabled."""
    live = _Live(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(live.poll_forever()) if settings.live.enabled else None
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(
        title="money-printer-research",
        description="Clean football match tables and Elo-model H/D/A predictions.",
        lifespan=lifespan,
    )
    tables = _Tables(settings.clean.out_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "tables": len(tables.names())}

    @app.get("/tables")
    def list_tables() -> list[str]:
        return tables.names()

    @app.get("/tables/{name}")
    def get_table(
        name: str,
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        df = tables.read(name)
        return {
            "name": name,
            "total": df.height,
            "offset": offset,
            "limit": limit,
            "columns": df.columns,
            "rows": _rows(df.slice(offset, limit)),
        }

    @app.get("/predictions/seasons")
    def prediction_seasons() -> list[str]:
        return tables.seasons()

    @app.get("/predictions")
    def predictions(
        season: str, team: str | None = None, dixon_coles: bool = True
    ) -> list[dict[str, Any]]:
        pred = tables.predictions(season, dixon_coles)
        return _rows(pred if team is None else _involving(pred, team))

    @app.get("/predictions/metrics")
    def prediction_metrics(season: str, dixon_coles: bool = True) -> dict[str, Any]:
        return evaluate(tables.predictions(season, dixon_coles))

    @app.get("/upcoming")
    def upcoming(league: str | None = None, team: str | None = None) -> list[dict[str, Any]]:
        pred = _read_output(settings.modeling.predictions_file, "predict")
        if league is not None:
            pred = pred.filter(pl.col("league") == league)
        if team is not None:
            pred = _involving(pred, team)
        return _rows(pred)

    @app.get("/backtest")
    def backtest(
        min_ev: Annotated[float, Query(ge=-1)] = 0.0, league: str | None = None
    ) -> dict[str, Any]:
        table = _read_output(settings.modeling.backtest_file, "backtest")
        leagues = sorted(table["league"].unique())
        if league is not None:
            table = table.filter(pl.col("league") == league)
        return {
            "leagues": leagues,
            "games": table.height,
            "log_loss": log_loss_vs_market(table) if table.height else None,
            "total": _rows(summary(table, min_ev))[0],
            "by_season": _rows(summary(table, min_ev, ["season_start"])),
            "cumulative_profit": _rows(cumulative_profit(table, min_ev)),
            "walk_forward": _rows(walk_forward_thresholds(table)) if table.height else [],
        }

    @app.get("/analysis")
    def analysis(league: str | None = None) -> dict[str, Any]:
        path = settings.modeling.analysis_file
        if not path.is_file():
            raise HTTPException(404, f"no {path}; run `uv run report` first")
        result = json.loads(path.read_text(encoding="utf-8"))
        strategies = result["strategies"]
        strategies["leagues"] = sorted({row["league"] for row in strategies["scores"]})
        if league is not None:
            for key in ("scores", "walk_forward"):
                strategies[key] = [row for row in strategies[key] if row["league"] == league]
        return result

    @app.get("/live")
    def live_games(league: str | None = None) -> list[dict[str, Any]]:
        today = datetime.now(UTC).date().isoformat()
        return live.store.games(since=today, league=league)

    @app.get("/live/{espn_id}")
    def live_game(espn_id: str) -> dict[str, Any]:
        game = live.store.game(espn_id)
        if game is None:
            raise HTTPException(404, f"no live game {espn_id!r}; see GET /live")
        return game

    @app.get("/teams")
    def list_teams(league: str | None = None) -> list[dict[str, Any]]:
        matches = tables.read(MATCHES_TABLE)
        if league is None:
            return _rows(team_tables.teams(matches))
        season = team_tables.latest_season(matches, league)
        if season is None:
            raise HTTPException(404, f"no games for league {league!r}")
        return _rows(team_tables.league_table(matches, league, season))

    @app.get("/teams/{team}")
    def team(team: str) -> dict[str, Any]:
        stats = team_tables.team_stats(tables.read(MATCHES_TABLE), tables.read(ELO_TABLE), team)
        if stats is None:
            raise HTTPException(404, f"no games for team {team!r}; see GET /teams")
        return stats

    # Last, so the API routes above take precedence: the pages and their CSS and JS.
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="pages")

    return app


def exec_api() -> None:
    """Entry point `api`: serve create_app() on settings.api_settings.host and port."""
    import uvicorn

    api = default_settings.api_settings
    uvicorn.run(create_app(), host=api.host, port=api.port)
