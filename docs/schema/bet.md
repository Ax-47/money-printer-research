# Bet

The backtest: each game's odds, the model's probabilities and the pick it would bet,
and how a strategy over those picks did season by season. See
[`backtest`](../cli.md#backtest).

| Stage | Model | Used for |
| --- | --- | --- |
| Backtest | `BacktestCols` | `backtest.csv`, one row per game with odds |
| Backtest | `ThresholdSeasonCols` | `walk_forward_thresholds`, one row per season |

## BacktestCols

One row per game from `modeling.backtest_from` that football-data has odds for.
`pick` is the outcome with the highest expected value, `ev` that value and `profit`
what 1 unit on it returned, so a strategy "bet every pick with `ev` above t" is a
filter on this table.

<!-- columns: money_printer_research.schema.BacktestCols -->

## ThresholdSeasonCols

Choosing the threshold after seeing the results flatters them. Here each season is
bet at the threshold that made the most on the seasons before it only, as someone
betting live would have to choose.

<!-- columns: money_printer_research.schema.ThresholdSeasonCols -->
