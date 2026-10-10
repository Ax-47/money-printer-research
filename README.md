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

Paths, leagues and seasons live in `settings.toml`; any value can be overridden
with an environment variable such as `UNDERSTAT__START_YEAR=2018`.

## Pipeline

```bash
uv run collect_data        # Understat schedule and player stats, Transfermarkt tables -> data/raw
uv run clean_data          # cleaned tables and pre-match features -> data/clean
uv run feature_importance  # permutation importance for the H/D/A model
uv run notebook            # JupyterLab at the project root
uv run check_clean         # validate every CSV in data/clean against its schema
```

`collect_data` skips files that already exist, so it can be stopped and resumed.

## Development

```bash
make check  # lock file, ruff, pyright, pre-commit
make test   # pytest with coverage
make help   # every target
```

---

Repository initiated with [tiefenthaler/uv-datascience-project-template](https://github.com/tiefenthaler/uv-datascience-project-template).
