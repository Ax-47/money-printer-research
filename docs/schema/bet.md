# Bet

Bookmaker prices and results from football-data.co.uk, the backtest's picks, and how a
strategy over those picks did season by season. See [`backtest`](../cli.md#backtest).

| Stage | Model | Used for |
| --- | --- | --- |
| Odds | `OddsCols` | Every published price, one row each (`analysis.odds.read_prices`) |
| Odds | `ResultCols` | Every game's full-time score, to settle bets |
| Backtest | `BacktestCols` | `backtest.csv`, one row per game with odds |
| Backtest | `ThresholdSeasonCols` | `walk_forward_thresholds`, one row per season |

## OddsCols

football-data publishes one column per price, such as `PSCH` (Pinnacle, closing, home
win) or `MaxCAHA` (best price across bookmakers, closing, away side of the Asian
handicap). `read_prices` turns each into a row, so code asks for a source, timing and
market instead of a column name.

What exists depends on the season:

| Prices | Seasons |
| --- | --- |
| Pinnacle 1X2, early and closing | all |
| Max and average across bookmakers, 1X2 / AH / O/U 2.5, early and closing | 2019/20 on |
| Max and average (Betbrain), 1X2 / AH / O/U 2.5, early only | to 2018/19 |
| Betfair exchange | 2024/25 on |

"Early" is football-data's pre-closing price, collected a day or two before kick-off. It
is not the true opening price, which is not published.

<!-- columns: money_printer_research.schema.OddsCols -->

## ResultCols

<!-- columns: money_printer_research.schema.ResultCols -->

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
