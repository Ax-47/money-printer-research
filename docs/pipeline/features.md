# 3. Features

`extract_features` in `money_printer_research.clean_data` builds the pre-match
features from the [clean tables](clean.md) and joins them onto the matches.
`uv run clean_data` runs it after `clean` and writes six more CSVs.

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
| `missing.csv` | [`MissingCols`](../schema/match.md) | `features.missing.missing_table` |
| `model_frame.csv` | `model_frame_schema` | `features.impute.model_frame` |

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
its source has no value for the game; `missing.csv` says why (see
[Missing values](#missing-values)). The result is `match_features.csv`, the
modelling table.

## Missing values

`match_features.csv` keeps its nulls and gains no columns for them. Instead
`missing_table` writes `missing.csv`, a table of its own with one row per null cell:
the match (`game_id`), the `column`, the `side` (`h`, `a`, or null for a column of
the whole match) and the `reason`:

| Reason | Null cells |
| --- | --- |
| `no_lineup` | A side's fatigue and player slots, when Understat has no starters for it |
| `short_lineup` | A player slot left empty, when fewer than 11 starters are recorded |
| `too_few_minutes` | A starter's `xg90`, `xa90` and `form90`, under 270 previous minutes |
| `no_schedule` | A side's schedule features, when its club has no Transfermarkt game |
| `no_next_game` | `days_to_next_all`, for a club's last game in the Transfermarkt data |
| `no_tm_game` | The Transfermarkt columns, when the match has no Transfermarkt game |
| `no_formation` | A side's formation columns, when its Transfermarkt game has no line-up |
| `plain_shape` | `home_variant` / `away_variant`, for a shape with no variant such as "4-2-3-1" |
| `other` | None of these, such as a formation that did not parse |

Every null cell of `match_features.csv` has exactly one row, so the two tables
always agree. The nulls stay null. Filling them with a mean or median of the table
would leak later seasons into earlier ones, and a constant such as 0 would read as a
real value: a newcomer is not a player with no expected goals. Gradient-boosted trees,
as in `feature_importance`, take nulls as they are; a model that cannot should fill
them inside its own training fold, and can join `missing.csv` to know why.

The one gap filled at the source is `rest_days_all` for a club's first game,
which gets the 30-day cap, since the cap already means "a month or more".

### Model frame

`model_frame.csv` is `match_features.csv` with the gaps whose meaning is known
handled, so a model can read it as it is. `match_features.csv` itself stays as it
was. Each step answers one reason, as checked on the real data:

| Reason | In `model_frame.csv` |
| --- | --- |
| `no_lineup` | The match is dropped: two matches with no starters on either side |
| `no_formation` | The team's formation from its previous match, with `h_formation_imputed` / `a_formation_imputed` set; null if the team has none earlier. Most are La Liga 2018/19's first rounds |
| `no_next_game` | `days_to_next_all` = 30, the cap: the club's last game before the summer break |
| `plain_shape` | `home_variant` / `away_variant` = `"none"` |
| `too_few_minutes`, `short_lineup` | Kept null |

The first season (2014/15) is dropped last, as warm-up: every player and team starts
with no history, and about 18% of its starter slots are `too_few_minutes`, against
3-8% later. The formation fill runs before, so 2015/16 can use it. It only looks
back: a team's most common formation would read its later games.

`uv run clean_data` ends by printing `reason_counts(missing)`: the null cells and
the matches they touch, per reason. `null_report` counts nulls per feature group
instead:

```python
import polars as pl
from money_printer_research.features import null_report, reason_counts

missing = pl.read_csv("data/clean/missing.csv")
print(reason_counts(missing))
```
