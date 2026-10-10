# Getting started

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
make install        # uv sync --all-groups: runtime, dev and docs dependencies
make install-hooks  # pre-commit hooks
```

`uv sync` alone installs the runtime and `dev` groups; the docs need
`--all-groups` (or `--group docs`).

## Settings

Paths, leagues and seasons live in `settings.toml` at the project root. See
[Configuration](configuration.md).

## Run the pipeline

```bash
uv run collect_data   # Understat schedule and player stats, Transfermarkt tables
uv run clean_data     # cleaned tables and pre-match features -> data/clean
uv run check_clean    # validate every CSV in data/clean
```

`collect_data` skips files that already exist, so it can be stopped and resumed.
The first run downloads the whole Transfermarkt database and one Understat page
per match, so it takes a while. See [Data pipeline](pipeline/index.md) for what
each step writes.

## Development

```bash
make check        # lock file, ruff, pyright, pre-commit
make test         # pytest with coverage
make docs-serve   # this site, live at http://127.0.0.1:8000
make docs-build   # build the site; fails on any warning (what CI runs)
make help         # every target
```
