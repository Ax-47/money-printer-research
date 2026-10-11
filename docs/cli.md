# Commands

Every command is a console script declared in `pyproject.toml` under
`[project.scripts]`. Run them with `uv run <name>`. They take no arguments;
everything they read comes from the [settings](configuration.md).

| Command | Entry point | What it does |
| --- | --- | --- |
| `collect_data` | `data_collection.collect_data` | Download the Understat schedule, Understat player stats and the Transfermarkt tables. See [Collect](pipeline/collect.md). |
| `clean_data` | `clean_data.exec_clean_data` | Clean the collected files and build the pre-match features; write nine CSVs. See [Clean](pipeline/clean.md) and [Features](pipeline/features.md). |
| `check_clean` | `schema.check_clean` | Validate every CSV in the clean output folder against its schema. |
| `feature_importance` | `analysis.exec_feature_importance` | Forward feature selection for the home / draw / away model on `model_frame.csv`, walk-forward by season. |
| `modeling` | `analysis.exec_modeling` | Compare home / draw / away models on the selected features, check the baseline's calibration, score them on the held-out seasons, then save the logistic model. |
| `predict` | `analysis.exec_predict` | H/D/A probabilities for every unplayed fixture in the schedule, from the saved model. |
| `backtest` | `analysis.exec_backtest` | Bet the model's best-EV outcome against historical bookmaker odds, season by season, and report profit and ROI. |
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

Reads `model_frame.csv` and selects feature groups for a multinomial logistic
regression by greedy forward selection from Elo: each step adds the group that
lowers the walk-forward log loss (seasons 2021 to 2023, each predicted by a model
fitted on the seasons before it) the most, until no group gains `MIN_GAIN`. It
then scores Elo alone, the selection and every group once on the held-out
seasons from 2024, and says so if the selection differs from `SELECTED`.

## `modeling`

Fits each model in `analysis.modeling.MODELS` on the `SELECTED` features, on the
same walk-forward seasons, and prints log loss, ranked probability score, Brier
score and accuracy; then the logistic baseline's calibration by probability
decile; then every model once on the held-out seasons from 2024. Last, it refits the
logistic model on every season and saves it with joblib to `modeling.out_file`
(`data/models/hda_logistic.joblib`); `analysis.modeling.load_model` reads it back and
its `predict` takes `model_frame` rows. joblib runs code from the file, so load only
models you saved yourself.

## `predict`

Loads the model `modeling` saved and predicts every fixture the Understat schedule
holds but has not seen played (`is_result` false), then writes `game_id`, `date`,
`league`, both teams, `p_H`, `p_D` and `p_A` to `modeling.predictions_file`
(`data/predictions/upcoming.csv`). Run `modeling` first.

Each fixture's features come from `match_features.csv`, using only what is known
before kick-off:

- Elo: each team's rating after its last played match; promoted teams start a new
  season the way `clean_data` starts them.
- Games in the last 14 days: league games only (played ones and earlier fixtures), so
  a cup or European game in that window is missed.
- Player slots: the line-up is unknown before kick-off, so a team gets the starters of
  its last played match and their values from then: the likely XI, one game old.

The schedule covers the seasons from `understat.start_year` up to, not including,
`understat.end_year`. To predict the season in progress, set `end_year` past it
(`UNDERSTAT__END_YEAR=2027` for 2026-27) and run `collect_data` and `clean_data`.
`collect_data` skips a schedule file that already holds every season, so delete
`data/raw/xg_schedule.csv` to pick up newer results and fixtures.

## `backtest`

Downloads football-data.co.uk's 1X2 odds for each league and season to
`modeling.odds_dir` (once; the latest season is fetched again each run), predicts
every season from `modeling.backtest_from` with the logistic baseline fitted only on
the seasons before it, and prices each game with Pinnacle's closing odds (falling back
to Pinnacle's opening price, the market average at closing, then Bet365).

Each game's pick is the outcome with the highest expected value of a 1-unit stake,
`p * odds - 1`. It prints the model's and the bookmaker's log loss and, for a few EV
thresholds, the bets, hit rate, profit and ROI of betting every pick above it, and
writes every game to `modeling.backtest_file` (`data/predictions/backtest.csv`).
Needs `data/clean/model_frame.csv`, as `modeling` does. The web page at
`/backtest.html` shows the same with a threshold slider.

football-data names teams differently from Understat; each name is matched to the
team it shares the most match days with, so there is no alias list to keep.

## `notebook`

Starts JupyterLab with the project root as the notebook directory. The analysis
notebooks live in `notebooks/`.
