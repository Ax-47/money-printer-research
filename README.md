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
uv run collect_data        # Understat schedule and player stats, Transfermarkt tables -> data/raw
uv run clean_data          # cleaned tables and pre-match features -> data/clean
uv run feature_importance  # permutation importance for the H/D/A model
uv run notebook            # JupyterLab at the project root
uv run check_clean         # validate every CSV in data/clean against its schema
uv run show_config         # print the settings in effect
```

`collect_data` skips files that already exist, so it can be stopped and resumed.

## API

```bash
uv run api  # http://localhost:8000/docs
```

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
