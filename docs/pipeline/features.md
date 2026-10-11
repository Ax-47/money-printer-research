# 3. Features

`extract_features` in `money_printer_research.clean_data` builds the pre-match
features from the [clean tables](clean.md) and joins them onto the matches.
`uv run clean_data` runs it after `clean` and writes five more CSVs.

Every feature uses only what was known at kick-off: rolling windows exclude the
current match, and Elo ratings are taken before the match is played. The two
exceptions are noted under [Schedule](#schedule).

| Output | Schema | Built by |
| --- | --- | --- |
| `elo.csv` | [`EloCols`](../schema/match.md) | `features.get_elo` |
| `fatigue.csv` | [`TeamFatigueCols`](../schema/match.md) | `features.get_fatigue` |
| `schedule.csv` | [`TeamScheduleCols`](../schema/match.md) | `features.team_schedule_features` |
| `players.csv` | `PLAYER_SLOTS` | `features.get_player_features` |
| `match_features.csv` | [`MatchFeaturesCols`](../schema/match.md) | `clean_data.join.join_features` |

The first four are computed independently, so if several fail, every error comes
back together.

## Elo

`get_elo` replays every league match in date order and records each side's
rating before kick-off as `elo_home` and `elo_away`, with
`elo_diff = elo_home - elo_away`.

- Update size `k = 15`, home advantage of 40 points in the expectation, base
  rating 1500. These were tuned by walk-forward log loss; the surface is flat.
- The update is scaled by goal difference: 1 for one goal, 1.5 for two, then
  `(11 + gd) / 8`.
- Promoted teams start each season at the mean end-of-season rating of the teams
  that went down, which keeps the league average stable.
- Every team starts level in 2014/15, so treat that season as warm-up.

`features.elo` also holds a double Poisson match model driven by `elo_diff`
(`walk_forward`, `evaluate`), with an optional Dixon-Coles correction. It
predicts each season with a model fitted only on earlier seasons.

## Fatigue

`get_fatigue` works per player, then summarises each side's starters.

For every player and match, `player_fatigue` computes the load before kick-off:
rest days, matches and minutes in the last 7 days, minutes in the last 28 days,
and consecutive team games played. These combine into a `fatigue_score` from 0
to 100, with hand-set weights:

| Weight | Term (each scaled to 0 to 1) |
| --- | --- |
| 0.35 | Minutes in the last 7 days (180 is full) |
| 0.25 | Short rest |
| 0.20 | Consecutive games played |
| 0.20 | Acute to chronic workload ratio above 1 |

`team_fatigue` then gives each match the mean starter score per side
(`h_fatigue`, `a_fatigue`), the starters' summed 7-day minutes, and
`fatigue_diff`. A side with no recorded starters gets nulls.

!!! note
    Understat covers league matches only, so player rest is overstated for clubs
    that also play cups or European games. The schedule features below cover that
    gap at team level.

## Schedule

`club_calendar` builds every club's full calendar from Transfermarkt
(`match_leagues`, all competitions), and `team_schedule_features` attaches it to
each Understat match as `h_*` and `a_*` columns:

- `rest_days_all`: days since the previous game in any competition, capped at 30.
  A club's first game in the data has no previous one and gets 30.
- `games_7d_all`, `games_14d_all`: games in the previous 7 and 14 days.
- `non_league_14d`: cup, European or other games in the previous 14 days.
- `prev_europe_away`: the previous game was an away European tie.
- `days_to_next_all`, `next_is_europe`: look-ahead features, from fixtures
  already scheduled. `days_to_next_all` uses the date the next game was actually
  played, which for a postponed game was not known at kick-off.

A side whose club has no matched Transfermarkt game gets nulls.

## Players

`get_player_features` gives each starter a form line and spreads the eleven
starters of each side into slot columns, ordered by line-up position from the
goalkeeper forward. For each side (`h`, `a`), slot (1 to 11) and feature, the
column is `<side>_player<slot>_<feature>`, for example `h_player1_xg90`:

| Feature | Meaning |
| --- | --- |
| `xg90`, `xa90` | xG and xA per 90 minutes over the player's previous 10 matches |
| `form90` | xG plus xA per 90 over the same window: expected goal involvement |
| `prev_min` | Minutes played over the same window |
| `fatigue_score` | The player's fatigue score (see [Fatigue](#fatigue)) |

That makes 110 columns. Per-90 rates are null below 270 previous minutes, so a
newcomer's numbers are not noise.

## Join

`join_features` left-joins Elo, fatigue, schedule and the player slots onto
`cleaned_matches`, one to one, so every match keeps its row. Elo has no
`game_id`, so it joins on the match date and both teams. A feature is null where
its source has no value for the game, and flag columns say which source was
missing (see [Missing values](#missing-values)). The result is
`match_features.csv`, the modelling table.

## Missing values

`features.missing` decides what happens to the gaps in `match_features.csv`.
A feature is null for one of four reasons:

| Reason | Null columns | Flag |
| --- | --- | --- |
| Understat has no line-up for a side | that side's fatigue and player slots | `h_has_lineup`, `a_has_lineup` |
| A side's club has no matched Transfermarkt game | that side's schedule features | `h_has_schedule`, `a_has_schedule` |
| A starter has under 270 previous minutes | his `xg90`, `xa90` and `form90` slots | `h_rated_starters`, `a_rated_starters` |
| A club's last game in the Transfermarkt data | `days_to_next_all` | none |

The nulls stay null. Filling them with a mean or median of the table would leak
later seasons into earlier ones, and a constant such as 0 would read as a real
value: a newcomer is not a player with no expected goals. The flags give a model
the reason instead. `has_lineup` and `has_schedule` are true when that source had
the side; `rated_starters` counts the starters, 0 to 11, with per-90 rates.
Gradient-boosted trees, as in `feature_importance`, take nulls as they are; a
model that cannot should fill them inside its own training fold.

The one gap filled at the source is `rest_days_all` for a club's first game,
which gets the 30-day cap, since the cap already means "a month or more".

`uv run clean_data` ends by printing what is left, per feature group, from
`null_report`:

```python
import polars as pl
from money_printer_research.features import null_report

print(null_report(pl.read_csv("data/clean/match_features.csv", infer_schema_length=None)))
```
