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
| `end_year` | `2026` | exclusive: `2026` makes 2025-26 the last season; must be after `start_year` |
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

Unknown keys inside these sections are errors. Other top-level sections, such as
`[api_settings]`, belong to other parts of the project and are not read here.

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
