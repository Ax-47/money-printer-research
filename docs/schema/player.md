# Player

One row per player per match, from Understat. Key: `(game_id, player_id)`.

| Stage | Model | Used for |
| --- | --- | --- |
| Raw | `UnderstatPlayerRaw` | `<league>/<season>.parquet` as collected |
| Selected | `UnderstatPlayer` | The raw columns the clean step reads |
| Clean | `PlayerStatCols` | `player_stat.csv` |
| Features | `FatigueInputCols` | Player stats with the match date and side |
| Features | `PlayerFatigueCols` | Pre-match load per player per match |

## UnderstatPlayerRaw

<!-- columns: money_printer_research.schema.UnderstatPlayerRaw -->

## PlayerStatCols

<!-- columns: money_printer_research.schema.PlayerStatCols -->

## PlayerFatigueCols

`FatigueInputCols` is `PlayerStatCols` plus `match_date` and `is_home`;
`PlayerFatigueCols` adds the pre-match load. Only earlier matches count.

<!-- columns: money_printer_research.schema.PlayerFatigueCols -->
