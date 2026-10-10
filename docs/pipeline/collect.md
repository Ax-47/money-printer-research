# 1. Collect

`uv run collect_data` downloads every source the clean step reads. It runs three
collectors from `money_printer_research.data_collection`, fastest first.

Every collector follows the same two rules:

- **Resumable.** A file that already exists is skipped, so the command can be
  stopped and started again.
- **Checked before saving.** A frame is written to `<name>.part.<ext>`, read back
  and validated against its full raw schema (`schema.check_collected`), and only
  then renamed to its final name. A source that changed shape fails at download
  time and leaves no file behind, so the next run tries again.

## Understat schedule

`collect_schedule` reads the schedule of every configured league and season in
one soccerdata call and writes a single CSV (`data/raw/xg_schedule.csv` with the
shipped settings).

- One row per fixture, played or not, with goals, xG and the Understat `game_id`.
- Checked against [`UnderstatScheduleRaw`](../schema/match.md).
- Skipped when the file already holds every configured season.

## Understat player stats

`collect_player_stats` writes one Parquet file per league and season:
`<out_dir>/<league>/<season>.parquet`, for example
`data/raw/player_stats/ENG-Premier League/2014-15.parquet`.

- One row per player per match: minutes, position, goals, xG, xA and more.
- Checked against [`UnderstatPlayerRaw`](../schema/player.md).
- Understat serves one page per match, so this is the slow step. soccerdata
  caches pages under the Understat cache folder.
- When one match has no rosters, the whole season read fails; the collector
  then falls back to reading match by match and prints the matches it skipped.
- A league and season that fails is printed as `FAILED` and skipped; the next
  run retries it.

## Transfermarkt

`collect_transfermarkt` downloads the dcaribou/transfermarkt-datasets DuckDB
file once, then exports three tables to Parquet in the Transfermarkt output
folder:

| File | Rows kept | Raw schema |
| --- | --- | --- |
| `competitions.parquet` | Every competition | [`TransfermarktCompetitionsRaw`](../schema/competition.md) |
| `games.parquet` | Every game, in any competition, involving a club that played in the configured leagues, from the first season on | [`TransfermarktGamesRaw`](../schema/match.md) |
| `appearances.parquet` | Every appearance by a player who played in those leagues, from one season earlier | none yet |

Appearances keep one extra season so rolling windows at the start of the first
season have history.

- The download streams to `<file>.part` and is renamed when complete, so a
  broken download never looks finished.
- The dataset stopped updating in July 2026, so both the download and the
  export are skipped once their files exist. Delete them to force a refresh.

## Where the files go

| Source | Default location |
| --- | --- |
| Understat schedule | `data/raw/xg_schedule.csv` |
| Understat player stats | `data/raw/player_stats/<league>/<season>.parquet` |
| Understat page cache | `data/Understat/` |
| Transfermarkt database | `data/Transfermarkt/transfermarkt-datasets.duckdb` |
| Transfermarkt tables | `data/raw/transfermarkt/{competitions,games,appearances}.parquet` |

The locations, leagues and seasons come from the settings; see
[Configuration](../configuration.md).
