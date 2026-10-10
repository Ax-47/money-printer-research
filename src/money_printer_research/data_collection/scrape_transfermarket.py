"""Collect the Transfermarkt dataset (dcaribou/transfermarkt-datasets).

Downloads the single DuckDB file once, then exports only the rows the pipeline
needs to <out_dir>/{games,appearances,competitions}.parquet: every game
involving a club from the configured leagues since `first_season`, and every
appearance by a player who played in those leagues (cups, Europe and
national-team tournaments included). The dataset stopped updating in July 2026,
so both steps are skipped once their files exist; delete them to force a refresh.

Each table is written to <name>.part.parquet and checked against its raw schema
before it replaces <name>.parquet, so a table that fails the check leaves no file
and the next run exports it again.
"""

import urllib.request
from pathlib import Path

import duckdb

from money_printer_research.config import Settings
from money_printer_research.schema.raw import LEAGUE_INFO, MATCH_INFO, RawSchema, check_collected

TABLES = ("competitions", "games", "appearances")
# Raw schema of each table the clean pipeline reads; appearances has none yet.
SCHEMAS: dict[str, RawSchema] = {"competitions": LEAGUE_INFO, "games": MATCH_INFO}

# Transfermarkt stores season as text; compare its first four characters as a year.
SEASON_YEAR = "TRY_CAST(left(CAST(games.season AS VARCHAR), 4) AS INTEGER)"

MB = 2**20
CHUNK_SIZE = MB
PROGRESS_EVERY = 50 * MB


def _print_progress(done: int, total: int) -> None:
    """Print bytes downloaded so far, in MB, out of `total` when the server sent it."""
    of_total = f"/{total / MB:,.0f}" if total else ""
    print(f"  {done / MB:,.0f}{of_total} MB", flush=True)


def _download(url: str, dest: Path) -> None:
    """Stream to <dest>.part and rename at the end, so a broken download never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    # The host rejects urllib's default User-Agent with 403; curl's works.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(CHUNK_SIZE):
            out.write(chunk)
            done += len(chunk)
            # True once per PROGRESS_EVERY bytes, in the chunk that crosses the mark.
            if done % PROGRESS_EVERY < CHUNK_SIZE:
                _print_progress(done, total)
    tmp.replace(dest)


def _create_scope_tables(con: duckdb.DuckDBPyConnection, in_scope: str) -> None:
    """Temp tables of the clubs and players that played an in-scope league game."""
    con.execute(f"""
        CREATE TEMP TABLE scope_clubs AS
        SELECT DISTINCT club_id FROM club_games JOIN games USING (game_id)
        WHERE {in_scope}
    """)
    con.execute(f"""
        CREATE TEMP TABLE scope_players AS
        SELECT DISTINCT player_id FROM appearances JOIN games USING (game_id)
        WHERE {in_scope}
    """)


def _export_queries(first_season: int) -> dict[str, str]:
    """Table name -> the SELECT that exports it; reads the scope_* temp tables."""
    return {
        "competitions": "SELECT * FROM competitions",
        "games": f"""
            SELECT * FROM games
            WHERE {SEASON_YEAR} >= {first_season}
              AND (home_club_id IN (SELECT club_id FROM scope_clubs)
                   OR away_club_id IN (SELECT club_id FROM scope_clubs))
        """,
        "appearances": f"""
            SELECT appearances.* FROM appearances JOIN games USING (game_id)
            WHERE {SEASON_YEAR} >= {first_season - 1}
              AND appearances.player_id IN (SELECT player_id FROM scope_players)
        """,
    }


def _copy_checked(con: duckdb.DuckDBPyConnection, name: str, sql: str, out_dir: Path) -> Path:
    """Write the query to <name>.parquet, after checking it against its raw schema if any."""
    path = out_dir / f"{name}.parquet"
    tmp = out_dir / f"{name}.part.parquet"
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET)")
    try:
        if name in SCHEMAS:
            check_collected(tmp, SCHEMAS[name])
    except ValueError:
        tmp.unlink()
        raise
    tmp.replace(path)
    return path


def export(db_file: Path, out_dir: Path, league_ids: list[str], first_season: int) -> None:
    """Write the filtered tables to parquet and print their row counts.

    league_ids: Transfermarkt competition ids of the leagues to keep, e.g. ["GB1", "ES1"].
    first_season: first season start year; appearances keep one extra season so
                  rolling windows at the start of `first_season` have history.
    """
    if not league_ids:
        raise ValueError("league_ids is empty: set transfermarkt.leagues in the config")
    out_dir.mkdir(parents=True, exist_ok=True)
    leagues = ", ".join(f"'{c}'" for c in league_ids)
    in_scope = f"games.competition_id IN ({leagues}) AND {SEASON_YEAR} >= {first_season}"

    con = duckdb.connect(str(db_file), read_only=True)
    try:
        _create_scope_tables(con, in_scope)
        for name, sql in _export_queries(first_season).items():
            path = _copy_checked(con, name, sql, out_dir)
            row = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()
            print(f"saved {name:13s} {row[0] if row else 0:>9,} rows", flush=True)
    finally:
        con.close()


def collect_transfermarkt(settings: Settings) -> None:
    """Download the Transfermarkt database once, then export the tables in scope."""
    cfg = settings.transfermarkt
    if all((cfg.out_dir / f"{table}.parquet").exists() for table in TABLES):
        print("skip transfermarkt", flush=True)
        return
    if not cfg.db_file.exists():
        print(f"downloading {cfg.url}", flush=True)
        _download(cfg.url, cfg.db_file)
    export(cfg.db_file, cfg.out_dir, list(cfg.leagues), cfg.first_season)
