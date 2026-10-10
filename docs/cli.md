# Commands

Every command is a console script declared in `pyproject.toml` under
`[project.scripts]`. Run them with `uv run <name>`. They take no arguments;
everything they read comes from the [settings](configuration.md).

| Command | Entry point | What it does |
| --- | --- | --- |
| `collect_data` | `data_collection.collect_data` | Download the Understat schedule, Understat player stats and the Transfermarkt tables. See [Collect](pipeline/collect.md). |
| `clean_data` | `clean_data.exec_clean_data` | Clean the collected files and build the pre-match features; write nine CSVs. See [Clean](pipeline/clean.md) and [Features](pipeline/features.md). |
| `check_clean` | `schema.check_clean` | Validate every CSV in the clean output folder against its schema. |
| `feature_importance` | `analysis.exec_feature_importance` | Permutation importance of the features for the home / draw / away model. |
| `notebook` | `notebook.exec_notebook` | Open JupyterLab at the project root with the project's environment. |
| `show_config` | `config.show_config` | Print the settings in effect and the file they came from. See [Configuration](configuration.md). |
| `api` | `api.exec_api` | Start the HTTP API. See [Docker and API](docker-api.md). |

Entry points are imported only when called, so importing the package (for
example `money_printer_research.clean_data` in a notebook) does not load the
scraping libraries.

## `collect_data`

Runs the three collectors, fastest first:

1. `collect_schedule`: one Understat request per league and season.
2. `collect_player_stats`: one Understat page per match.
3. `collect_transfermarkt`: one DuckDB download, then three Parquet exports.

A file that already exists is skipped, and a file is written only after it
passes its raw schema check, so an interrupted or failed run can simply be
started again. Failures for one league and season are printed as `FAILED` and
the run carries on with the next.

## `clean_data`

Runs [`clean`](pipeline/clean.md), writes its four CSVs, then runs
[`extract_features`](pipeline/features.md) and writes five more. If a step
fails, the command raises with that step's error; the clean CSVs stay on disk
even when feature extraction fails.

## `check_clean`

Reads each CSV listed in `schema.CLEAN_FILES` with the dtypes of its schema
(CSV loses some, such as `season`), validates it, and prints one line per file:

```text
ok      match_leagues     123,456 rows
MISSING data/clean/players.csv
FAIL    elo: ...
```

Exits with status 1 if any file is missing or fails.

## `feature_importance`

Trains a `HistGradientBoostingClassifier` on seasons before 2024 and scores it on
the rest, then reports how much the test log loss rises when each feature, or
each group of related features, is permuted.

!!! warning
    It currently reads `data/clean/matches.csv`, which `clean_data` no longer
    writes (it writes `match_features.csv`). This is tracked in
    [issue #13](https://github.com/Ax-47/money-printer-research/issues/13).

## `notebook`

Starts JupyterLab with the project root as the notebook directory. The analysis
notebooks live in `notebooks/`.
