# Configuration

Every path, league and season the pipeline uses is a typed setting in
`money_printer_research.config`. Values are checked when they are loaded, so a typo
or an impossible value stops the run with a message that names the setting instead
of failing halfway through a scrape.

## Where values come from

Highest priority first:

1. Arguments to `Settings(...)` or `load_settings(...)` (tests, notebooks).
2. Environment variables. Join the section and the key with `__`:
   `UNDERSTAT__START_YEAR=2018`, `CLEAN__OUT_DIR=/data/clean`.
   Lists and tables take JSON: `UNDERSTAT__LEAGUES='["ESP-La Liga"]'`.
3. A `.env` file in the working directory, with the same names as the environment variables.
4. `settings.toml`.
5. The defaults in `config.py` (shown in the tables below).

Sources are merged per key, so overriding `UNDERSTAT__START_YEAR` keeps every other
`[understat]` value from `settings.toml`.

## Which settings.toml

- The file named by `MPR_SETTINGS_FILE`, if set. A missing file is an error.
- Otherwise the first `settings.toml` found walking up from the working directory,
  then from the installed package.
- Otherwise none: defaults and environment variables only.

Relative paths resolve against the folder that holds `settings.toml` (the project
root), so `uv run clean_data` writes to the same place from any working directory.
Absolute paths are used as they are.

To see the values in effect and the file they came from:

```bash
uv run show_config
```

## Settings

### `[understat]`

| Key | Default | Notes |
| --- | --- | --- |
| `leagues` | the five big leagues | soccerdata names, at least one, no duplicates |
| `start_year` | `2014` | start year of the first season; Understat begins in 2014 |
| `end_year` | `2027` | exclusive: `2027` makes 2026-27, the current season, the last; must be after `start_year`. The current season's unplayed games are what `predict` predicts |
| `out_dir` | `data/raw/player_stats` | player stats, one parquet per match |
| `schedule_file` | `data/raw/xg_schedule.csv` | schedule with goals and xG |
| `cache_dir` | `data/Understat` | soccerdata cache |

### `[transfermarkt]`

| Key | Default | Notes |
| --- | --- | --- |
| `url` | transfermarkt-datasets DuckDB on R2 | downloaded once |
| `db_file` | `data/Transfermarkt/transfermarkt-datasets.duckdb` | local copy |
| `out_dir` | `data/raw/transfermarkt` | exported parquet tables |
| `leagues` | `GB1`, `ES1`, `L1`, `IT1`, `FR1` | Transfermarkt competition id to Understat league name; ids are upper-case letters and digits |
| `first_season` | `2014` | first season (start year) to export |

### `[clean]`

| Key | Default | Notes |
| --- | --- | --- |
| `out_dir` | `data/clean` | cleaned CSVs and features |

### `[modeling]`

| Key | Default | Notes |
| --- | --- | --- |
| `out_file` | `data/models/hda_logistic.joblib` | the H/D/A model `uv run modeling` fits on every season and saves |
| `predictions_file` | `data/predictions/upcoming.csv` | `uv run predict`'s output: H/D/A probabilities for every unplayed fixture |
| `odds_dir` | `data/raw/odds` | football-data.co.uk odds CSVs `uv run collect_data` downloads for `uv run backtest`: `<league>/<season>.csv` |
| `backtest_from` | `2017` | first season (start year) `uv run backtest` bets on; 2015 or later |
| `backtest_file` | `data/predictions/backtest.csv` | `uv run backtest`'s output: each game's odds, probabilities, pick and profit |
| `strategies_file` | `strategies.toml` | the pre-registered strategy suite `uv run strategies` runs |
| `report_file` | `data/report/report.html` | `uv run report`'s output: the one-page answer, rebuilt from the raw data |
| `analysis_file` | `data/report/analysis.json` | `uv run report`'s results as JSON, served by the API at `/analysis` |
| `learning_curve_file` | `data/report/learning_curve.html` | `uv run learning_curve`'s page (also written by `uv run modeling`): log loss by training games |

### `[api_settings]`

| Key | Default | Notes |
| --- | --- | --- |
| `host` | `127.0.0.1` | address `uv run api` binds to; `settings.toml` sets `0.0.0.0` so Docker can reach it |
| `port` | `8000` | 1 to 65535 |

### `[live]`

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `true` | `uv run api` polls ESPN for today's games, scores and starting XIs |
| `poll_seconds` | `60` | seconds between polls, at least 10 |
| `lineup_window_minutes` | `90` | how long before kick-off to start asking for the line-ups |
| `db_file` | `data/live.sqlite` | SQLite file of the games, line-ups and predictions |

Unknown keys inside these sections are errors.

## Errors

A bad value raises `ConfigError` listing every problem, with the environment variable
that sets it:

```text
Invalid settings (/path/to/settings.toml, env, .env):
  understat.start_year: Input should be a valid integer (got 'soon')  [env: UNDERSTAT__START_YEAR]
```

## In code

```python
from money_printer_research.config import Settings, UnderstatSettings, settings

settings.clean.out_dir  # loaded once, on first use
Settings(understat=UnderstatSettings(start_year=2020, end_year=2022))  # a variant for a test
```
