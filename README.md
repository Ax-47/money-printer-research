# money-printer-research

[![codecov](https://codecov.io/gh/Ax-47/money-printer-research/branch/main/graph/badge.svg)](https://codecov.io/gh/Ax-47/money-printer-research)
[![License](https://img.shields.io/github/license/Ax-47/money-printer-research)](https://img.shields.io/github/license/Ax-47/money-printer-research)

Collect, clean and model football match data from Understat and Transfermarkt.

- **Github repository**: <https://github.com/Ax-47/money-printer-research/>
- **Documentation** <https://Ax-47.github.io/money-printer-research/>

## Setup

```bash
make install        # uv sync --all-groups
make install-hooks  # pre-commit hooks
```

Paths, leagues and seasons live in `settings.toml` and are validated when loaded.
Any value can be overridden with an environment variable such as
`UNDERSTAT__START_YEAR=2018` or a `.env` file; `uv run show_config` prints the values
in effect. See [Configuration](docs/configuration.md).

## Pipeline

```bash
uv run collect_data        # Understat schedule and player stats, odds, Transfermarkt tables -> data/raw
uv run clean_data          # cleaned tables and pre-match features -> data/clean
uv run feature_importance  # feature selection for the H/D/A model
uv run modeling            # compare H/D/A models, save the chosen one -> data/models
uv run learning_curve      # log loss by training games -> data/report/learning_curve.html
uv run elo_sensitivity     # walk-forward log loss over Elo K x home_adv, 2005/06-2014/15
uv run predict             # H/D/A probabilities for unplayed fixtures -> data/predictions
uv run backtest            # bet the model's +EV picks against historical odds -> data/predictions
uv run market_test         # can a model forecast better than the market's own prices?
uv run strategies          # the pre-registered strategy suite in strategies.toml
uv run staking             # what flat, Kelly and martingale staking do to a bankroll
uv run practicality        # what collecting an edge would take: time, account limits, the exchange
uv run report              # all of the above as one page -> data/report/report.html
uv run notebook            # JupyterLab at the project root
uv run check_clean         # validate every CSV in data/clean against its schema
uv run show_config         # print the settings in effect
```

`collect_data` skips files that already exist, so it can be stopped and resumed.

## API

```bash
uv run api  # web page on http://localhost:8000, API docs on http://localhost:8000/docs
```

The page at `/` lists the `predict` command's H/D/A probabilities for the unplayed
fixtures, filterable by league and team, and turns bookmaker odds you type in into
expected value; run `uv run predict` first. `/backtest.html` shows the `backtest`
command's profit and ROI for any EV threshold. `/analysis.html` shows the `report`
command's answer to "can a bettor beat the market?": model vs market, every strategy
(filterable by league), staking and what collecting an edge would take; run
`uv run report` first. The data behind them is `GET /upcoming?league=...&team=...`,
`GET /backtest?min_ev=...&league=...` and `GET /analysis?league=...`.
Every page switches between English and Thai (ไทย) and between the four Catppuccin
flavors, and remembers both; type is IBM Plex Sans Thai, falling back to the OS's Thai
font offline.

Serves the tables in `data/clean` (`GET /tables`, `GET /tables/{name}`) and the Elo
model's walk-forward predictions (`GET /predictions?season=2024/25`,
`GET /predictions/metrics?season=2024/25`). Host and port are `[api_settings]` in
`settings.toml`.

## Docker

```bash
docker compose up -d api                        # API on http://localhost:8000
docker compose run --rm pipeline collect_data   # any pipeline command
docker compose run --rm pipeline clean_data
```

`./data` is mounted into the container, so collected and cleaned data stay on the host.

## Development

```bash
make check  # lock file, ruff, pyright, pre-commit
make test   # pytest with coverage
make help   # every target
```

---

Repository initiated with [tiefenthaler/uv-datascience-project-template](https://github.com/tiefenthaler/uv-datascience-project-template).
