# Club

Transfermarkt clubs: which Understat team each one is, and each club's full
calendar across every competition.

| Stage | Model | Used for |
| --- | --- | --- |
| Clean | `ClubMapCols` | `clubs.csv`, the club-to-team map |
| Features | `ClubCalendarCols` | Every club's games with schedule features, in memory |

## ClubMapCols

The map is learned from league fixtures, not from names. See
[Map clubs to teams](../pipeline/clean.md#map-clubs-to-teams).

<!-- columns: money_printer_research.schema.ClubMapCols -->

## ClubCalendarCols

One row per club per Transfermarkt game, any competition. `team_schedule_features`
attaches these to each Understat match as `h_*` and `a_*` columns
([`TeamScheduleCols`](match.md#teamschedulecols)).

<!-- columns: money_printer_research.schema.ClubCalendarCols -->
