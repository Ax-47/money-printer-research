"""Collect the Transfermarkt dataset (dcaribou/transfermarkt-datasets).

Downloads the single DuckDB file once, then exports only the rows the pipeline
needs to <out_dir>/{games,appearances,competitions}.parquet: every game
involving a club from the configured leagues since `first_season`, and every
appearance by a player who played in those leagues (cups, Europe and
national-team tournaments included). The dataset stopped updating in July 2026,
so both steps are skipped once their files exist; delete them to force a refresh.
"""

import urllib.request
from pathlib import Path

import duckdb

from money_printer_research.config import Settings

TABLES = ("competitions", "games", "appearances")

# Transfermarkt stores season as text; compare its first four characters as a year.
SEASON_YEAR = "TRY_CAST(left(CAST(games.season AS VARCHAR), 4) AS INTEGER)"


def _download(url: str, dest: Path) -> None:
    """Stream to <dest>.part and rename at the end, so a broken download never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    # The host rejects urllib's default User-Agent with 403; curl's works.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done, step = 0, 50 * 2**20
        while chunk := response.read(2**20):
            out.write(chunk)
            done += len(chunk)
            if done % step < 2**20:
                size = f"/{total / 2**20:,.0f}" if total else ""
                print(f"  {done / 2**20:,.0f}{size} MB", flush=True)
    tmp.replace(dest)


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
        queries = {
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
        for name, sql in queries.items():
            path = out_dir / f"{name}.parquet"
            con.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)")
            row = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()
            print(f"saved {name:13s} {row[0] if row else 0:>9,} rows", flush=True)
    finally:
        con.close()


def _collect_transfermarkt(settings: Settings) -> None:
    cfg = settings.transfermarkt
    if all((cfg.out_dir / f"{t}.parquet").exists() for t in TABLES):
        print("skip transfermarkt", flush=True)
        return
    if not cfg.db_file.exists():
        print(f"downloading {cfg.url}", flush=True)
        _download(cfg.url, cfg.db_file)
    export(cfg.db_file, cfg.out_dir, list(cfg.leagues), cfg.first_season)
