# Competition

Transfermarkt competitions, and how the pipeline groups them.

| Stage | Model | Used for |
| --- | --- | --- |
| Raw | `TransfermarktCompetitionsRaw` | `competitions.parquet` as collected |
| Selected | `TransfermarktCompetitions` | The raw columns the clean step reads |

## Competition types

`COMP_TYPES` lists the groups that `match_leagues.csv` puts every game in:

| `comp_type` | Meaning |
| --- | --- |
| `league` | One of the five configured leagues |
| `domestic_cup` | A domestic cup, the EFL Cup included |
| `europe` | A UEFA club competition or its qualifiers |
| `other` | Anything else, such as friendlies or other leagues |

## TransfermarktCompetitionsRaw

<!-- columns: money_printer_research.schema.TransfermarktCompetitionsRaw -->
