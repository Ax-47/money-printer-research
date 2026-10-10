# Match

One row per match. These models cover the Understat schedule and the
Transfermarkt games from download to the final modelling table.

| Stage | Model | Used for |
| --- | --- | --- |
| Raw | `UnderstatScheduleRaw` | `xg_schedule.csv` as collected |
| Raw | `TransfermarktGamesRaw` | `games.parquet` as collected |
| Selected | `UnderstatSchedule`, `TransfermarktGames` | The raw columns the clean step reads |
| Clean | `GameFormationCols` | Transfermarkt games with encoded formations |
| Clean | `ScheduleCols` | Cleaned Understat schedule, in memory |
| Clean | `MatchLeaguesCols` | `match_leagues.csv` |
| Clean | `CleanedMatchCols` | `cleaned_matches.csv` |
| Features | `EloInputCols`, `EloCols` | Elo input, and `elo.csv` |
| Features | `PredictionCols` | Output of the Elo match model |
| Features | `TeamFatigueCols` | `fatigue.csv` |
| Features | `TeamScheduleCols` | `schedule.csv` |
| Features | `PLAYER_SLOTS` | `players.csv` |
| Features | `MatchFeaturesCols` | `match_features.csv` |

## Raw

### UnderstatScheduleRaw

<!-- columns: money_printer_research.schema.UnderstatScheduleRaw -->

### TransfermarktGamesRaw

<!-- columns: money_printer_research.schema.TransfermarktGamesRaw -->

## Clean

### FormationCols

The encoded formation columns that `encode_formation` adds for each side.

<!-- columns: money_printer_research.schema.FormationCols -->

### ScheduleCols

<!-- columns: money_printer_research.schema.ScheduleCols -->

### MatchLeaguesCols

<!-- columns: money_printer_research.schema.MatchLeaguesCols -->

### CleanedMatchCols

<!-- columns: money_printer_research.schema.CleanedMatchCols -->

## Features

### EloCols

<!-- columns: money_printer_research.schema.EloCols -->

### PredictionCols

<!-- columns: money_printer_research.schema.PredictionCols -->

### TeamFatigueCols

<!-- columns: money_printer_research.schema.TeamFatigueCols -->

### TeamScheduleCols

<!-- columns: money_printer_research.schema.TeamScheduleCols -->

### Player slots

`players.csv` has `game_id` and 110 generated columns, one per side (`h`, `a`),
starter slot (1 to 11) and feature in `PLAYER_SLOT_FEATURES`
(`xg90`, `xa90`, `form90`, `prev_min`, `fatigue_score`). Column names come from
`slot_column`: `("h", 1, "xg90")` gives `h_player1_xg90`. Every slot column is
a nullable float; it is null when the match has no line-up data or fewer than
eleven starters were recorded. See [Features](../pipeline/features.md#players).

### MatchFeaturesCols

`match_features.csv` is `MatchFeaturesCols` plus the 110 player slot columns
above; `match_features_schema()` returns both together. The columns below
exclude the slots.

<!-- columns: money_printer_research.schema.MatchFeaturesCols -->
