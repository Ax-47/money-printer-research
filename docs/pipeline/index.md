# Data pipeline

The pipeline runs in three steps. Each step reads only what the previous one
wrote, and every frame is checked against a [schema](../schema/index.md) on the
way, so a source that changes shape fails at the step where it changed.

```text
collect_data   Understat schedule, Understat player stats, Transfermarkt tables
     │           -> raw CSV and Parquet files
clean_data     clean            -> match_leagues, clubs, cleaned_matches, player_stat
     │         extract_features -> elo, fatigue, schedule, players, match_features
check_clean    validate every CSV in the clean output folder
```

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| [1. Collect](collect.md) | `collect_data` | Understat, Transfermarkt | Raw CSV and Parquet files |
| [2. Clean](clean.md) | `clean_data` | Raw files | `match_leagues`, `clubs`, `cleaned_matches`, `player_stat` |
| [3. Features](features.md) | `clean_data` | Clean frames (in memory) | `elo`, `fatigue`, `schedule`, `players`, `match_features` |

`clean_data` runs steps 2 and 3 together and writes all nine CSVs to the clean
output folder (`data/clean` by default). `check_clean` validates them afterwards.

## Sources

- **Understat**, through [soccerdata](https://soccerdata.readthedocs.io/): the
  league schedule with goals and xG, and every player's stats for every league
  match. Understat covers league games only.
- **Transfermarkt**, from
  [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets):
  every club's full calendar (leagues, domestic cups, European and other games),
  the formations each side started with, and the list of competitions.

Understat gives the target and the per-player stats; Transfermarkt fills in the
games Understat does not see, which matter for rest and congestion. The two are
joined through a learned club-to-team map (see [Clean](clean.md#map-clubs-to-teams)).

## Error handling

Cleaning and feature steps return a `Result` (`Ok` or `Err`) from the
`cool_stuff` package instead of raising. Independent steps run together with
`all_ok`, which collects every failure into one `Err`; dependent steps stop at
the first `Err`. `clean_data` turns a final `Err` into an exception that names
the failing step.
