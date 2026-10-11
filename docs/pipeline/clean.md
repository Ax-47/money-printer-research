# 2. Clean

`clean` in `money_printer_research.clean_data` turns the collected files into
four clean tables. `uv run clean_data` runs it, writes each table as
`<name>.csv` to the clean output folder, then runs [feature extraction](features.md).

| Output | Schema | One row per |
| --- | --- | --- |
| `match_leagues.csv` | [`MatchLeaguesCols`](../schema/match.md) | Transfermarkt game, any competition |
| `clubs.csv` | [`ClubMapCols`](../schema/club.md) | Transfermarkt club |
| `cleaned_matches.csv` | [`CleanedMatchCols`](../schema/match.md) | Understat league match |
| `player_stat.csv` | [`PlayerStatCols`](../schema/player.md) | Player per Understat match |

The steps below run in this order. Each one checks its input and output against
a schema.

## Read

`clean_data.read` loads the raw files and validates them against their raw
schemas: the Understat schedule CSV, every player-stat Parquet file under the
Understat output folder, and Transfermarkt `games.parquet` and
`competitions.parquet`.

## Clean the Understat schedule

`clean_match_stat` (in `clean_data.match_stat`):

1. Keeps played matches only (`is_result` is true).
2. Adds `season_start`, the first year of the season: `1415` becomes `2014`.
3. Checks every team name against the alias table, then replaces it with the
   club's canonical name (see [Team names](#team-names)).
4. Validates the result as [`ScheduleCols`](../schema/match.md).

## Clean the Understat player stats

`clean_player_stat` (in `clean_data.player_stat`) adds `season_start`, checks and
normalises team names the same way, and validates the result as
[`PlayerStatCols`](../schema/player.md).

## Encode formations

`encode_formation` (in `clean_data.formation`) turns each Transfermarkt
formation string, such as `"4-2-3-1"` or `"4-4-2 double 6"`, into numbers for
both sides: the shape, the variant, the number of defenders, midfielders and
forwards, and the number of lines. A formation that does not parse gets null in
every encoded column. `unknown_formations` lists those for review.

## Join games to competitions

`join_match_w_league` (in `clean_data.match_leagues`) joins the Transfermarkt
games to their competitions and adds:

- `comp_type`: `league`, `domestic_cup`, `europe` or `other`. European ties are
  recognised by competition id, because Transfermarkt files some of them under
  type `other`; the EFL Cup is added by hand because the competitions table does
  not list it.
- `league`: the Understat league name for the five leagues, null otherwise.

The result is `match_leagues.csv`.

## Map clubs to teams

Transfermarkt identifies clubs by id and Understat by name, and the two spell
clubs differently. `map_club_w_team` (in `clean_data.map_club_team`) learns the
map from the fixtures themselves rather than from names: a Transfermarkt league
game and an Understat game are the same fixture when the league and score match
and the dates are at most one day apart. Each club id is paired with the
Understat team it matches most often.

The step fails, with the doubtful rows in the error, when:

- two club ids pair with the same team, or
- a club with at least 10 league games has fewer than 90% of them backing its
  pairing.

Either usually means a missing alias. The result is `clubs.csv`.

## Join Understat matches to Transfermarkt games

`join_match_w_stat` (in `clean_data.match`) left-joins each Understat match to
its Transfermarkt league game on league, season, home team and away team, using
the club map. It does not join on the date: postponed or resumed games can sit
days apart in the two sources. Transfermarkt columns get a `tm_` prefix; the
encoded formation columns keep their names.

Every Understat match keeps its row. The step fails if more than 1% of matches
find no Transfermarkt game. The result is `cleaned_matches.csv`.

## Team names

`clean_data.rename_team_names` gives every club one canonical lowercase name
shared by both sources. A raw name is lowercased, stripped of dots and trimmed,
then looked up in `ALIASES` (in `clean_data.aliases_team`).

`ALIASES` is also an allowlist: a name that is not in it fails the step instead
of silently splitting one club in two. When a new season brings new names,
print paste-ready lines and add them to the table:

```python
from money_printer_research.clean_data.rename_team_names import alias_block

print(alias_block(schedule))
```

## Checking a step did no damage

`clean_data.integrity` compares a step's input with its output on a key and counts
keys lost, added or duplicated, values changed, and values turned into null.
`notebooks/data_integrity.ipynb` runs it over every step, from the raw files to
`match_features.csv` and on to the CSVs on disk, and checks that Understat and
Transfermarkt agree on the score of every matched game.
