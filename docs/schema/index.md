# Schemas

Every frame the pipeline reads or writes has a
[pandera](https://pandera.readthedocs.io/) model in
`money_printer_research.schema`, with each column's dtype, checks and a
description. The models are grouped by entity, the thing one row describes:

| Entity | One row per | Page |
| --- | --- | --- |
| Match | Match | [Match](match.md) |
| Player | Player per match | [Player](player.md) |
| Club | Club, or club per game | [Club](club.md) |
| Competition | Transfermarkt competition | [Competition](competition.md) |

Inside each entity there is one module per pipeline stage, in order. An entity
has only the stages it needs.

| Module | Holds |
| --- | --- |
| `raw.py` | Every column of a collected file, as the source names it. Collection checks every column, so a source that changes shape fails at download time. |
| `selected.py` | Only the raw columns the clean step reads, each copied from `raw.py` with the same type and description. |
| `clean.py` | The cleaned tables written as CSV. |
| `features.py` | The pre-match features built from the clean tables. |

Everything is re-exported from the package, so any model can be imported as:

```python
from money_printer_research.schema import CleanedMatchCols, PlayerStatCols
```

## Using a model in code

Every column is declared as a pandera `FieldType`, so a column name can be used
directly in a polars expression and is typed as `str`:

```python
import polars as pl
from money_printer_research.schema import ScheduleCols

played.filter(pl.col(ScheduleCols.season) == 1415)
```

## Clean output files

`schema.CLEAN_FILES` maps each CSV that `clean_data` writes to its schema:

| File | Schema | Entity |
| --- | --- | --- |
| `match_leagues.csv` | `MatchLeaguesCols` | [Match](match.md#matchleaguescols) |
| `clubs.csv` | `ClubMapCols` | [Club](club.md#clubmapcols) |
| `cleaned_matches.csv` | `CleanedMatchCols` | [Match](match.md#cleanedmatchcols) |
| `player_stat.csv` | `PlayerStatCols` | [Player](player.md#playerstatcols) |
| `elo.csv` | `EloCols` | [Match](match.md#elocols) |
| `fatigue.csv` | `TeamFatigueCols` | [Match](match.md#teamfatiguecols) |
| `schedule.csv` | `TeamScheduleCols` | [Match](match.md#teamschedulecols) |
| `players.csv` | `PLAYER_SLOTS` | [Match](match.md#player-slots) |
| `match_features.csv` | `match_features_schema()` | [Match](match.md#matchfeaturescols) |

CSV loses some dtypes (a season such as `1516` comes back as an integer), so read
these files with `read_clean`, which applies the schema's dtypes and then
validates:

```python
from money_printer_research.config import settings
from money_printer_research.schema import read_clean

matches = read_clean(settings.clean.out_dir, "match_features")
```

`uv run check_clean` does the same for every file. See [Commands](../cli.md#check_clean).

## Raw and selected models

`schema.SELECTED` maps each raw model to its selected model. When the pipeline
starts using a new raw column, copy its block from the raw model into the
selected one.

The column tables on these pages are generated from the models when the site is
built, so they always match the code.
