"""The live database (settings.live.db_file): one SQLite file the poller writes and the API reads.

    games        one row per ESPN game: teams (Understat names once matched), state, score
    lineups      the starting XIs, each starter with the Understat player he was matched to
    predictions  H/D/A per game and source: "fixture" (predict's guessed XI) or "lineup"

Every write is an upsert, so polling the same game again only updates it.
"""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    espn_id     TEXT PRIMARY KEY,
    league      TEXT NOT NULL,
    kickoff     TEXT NOT NULL,
    state       TEXT NOT NULL,
    detail      TEXT NOT NULL,
    espn_home   TEXT NOT NULL,
    espn_away   TEXT NOT NULL,
    home_team   TEXT,
    away_team   TEXT,
    game_id     INTEGER,
    home_score  INTEGER,
    away_score  INTEGER,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lineups (
    espn_id     TEXT NOT NULL,
    side        TEXT NOT NULL,
    name        TEXT NOT NULL,
    position    TEXT NOT NULL,
    player_id   INTEGER,
    player      TEXT,
    prev_min    REAL,
    form90      REAL,
    PRIMARY KEY (espn_id, side, name)
);
CREATE TABLE IF NOT EXISTS predictions (
    espn_id     TEXT NOT NULL,
    source      TEXT NOT NULL,
    p_H         REAL NOT NULL,
    p_D         REAL NOT NULL,
    p_A         REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (espn_id, source)
);
"""

GAME_COLUMNS = (
    "espn_id league kickoff state detail espn_home espn_away home_team away_team game_id "
    "home_score away_score"
).split()
KEPT = {"home_team", "away_team", "game_id"}
LINEUP_COLUMNS = "espn_id side name position player_id player prev_min form90".split()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    """A connection to the live database; creates the file and tables on first use."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API reads from request threads while the poller writes from its own.
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        """Close the connection."""
        self.con.close()

    def upsert_game(self, game: dict[str, Any]) -> None:
        """Insert or update one game (GAME_COLUMNS)."""
        cols = ", ".join(GAME_COLUMNS)
        marks = ", ".join(f":{c}" for c in GAME_COLUMNS)
        # Once matched, keep the Understat names and game_id: the fixture leaves the
        # schedule's unplayed rows after the game, and the next poll finds no match.
        updates = ", ".join(
            f"{c} = COALESCE(excluded.{c}, games.{c})" if c in KEPT else f"{c} = excluded.{c}"
            for c in GAME_COLUMNS
            if c != "espn_id"
        )
        with self.con:
            self.con.execute(
                f"INSERT INTO games ({cols}, updated_at) VALUES ({marks}, :updated_at) "
                f"ON CONFLICT (espn_id) DO UPDATE SET {updates}, updated_at = excluded.updated_at",
                {**{c: game.get(c) for c in GAME_COLUMNS}, "updated_at": _now()},
            )

    def set_lineup(self, espn_id: str, starters: Iterable[dict[str, Any]]) -> None:
        """Replace the game's line-up."""
        rows = [{**{c: s.get(c) for c in LINEUP_COLUMNS}, "espn_id": espn_id} for s in starters]
        with self.con:
            self.con.execute("DELETE FROM lineups WHERE espn_id = ?", (espn_id,))
            self.con.executemany(
                f"INSERT INTO lineups ({', '.join(LINEUP_COLUMNS)}) "
                f"VALUES ({', '.join(f':{c}' for c in LINEUP_COLUMNS)})",
                rows,
            )

    def set_prediction(self, espn_id: str, source: str, p: dict[str, float]) -> None:
        """Insert or update one game's H/D/A from `source`."""
        with self.con:
            self.con.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (espn_id, source) "
                "DO UPDATE SET p_H = excluded.p_H, p_D = excluded.p_D, p_A = excluded.p_A, "
                "updated_at = excluded.updated_at",
                (espn_id, source, p["p_H"], p["p_D"], p["p_A"], _now()),
            )

    def has_lineup(self, espn_id: str) -> bool:
        """True once the game's line-up is stored."""
        row = self.con.execute("SELECT 1 FROM lineups WHERE espn_id = ? LIMIT 1", (espn_id,))
        return row.fetchone() is not None

    def has_prediction(self, espn_id: str, source: str) -> bool:
        """True once the game has a prediction from `source`."""
        row = self.con.execute(
            "SELECT 1 FROM predictions WHERE espn_id = ? AND source = ?", (espn_id, source)
        )
        return row.fetchone() is not None

    def games(self, since: str | None = None, league: str | None = None) -> list[dict[str, Any]]:
        """Games kicking off at or after `since` (ISO time), with both predictions, by kick-off."""
        sql = """
            SELECT g.*,
                   f.p_H AS fixture_p_H, f.p_D AS fixture_p_D, f.p_A AS fixture_p_A,
                   l.p_H AS lineup_p_H, l.p_D AS lineup_p_D, l.p_A AS lineup_p_A,
                   l.updated_at AS lineup_at,
                   EXISTS (SELECT 1 FROM lineups x WHERE x.espn_id = g.espn_id) AS has_lineup
            FROM games g
            LEFT JOIN predictions f ON f.espn_id = g.espn_id AND f.source = 'fixture'
            LEFT JOIN predictions l ON l.espn_id = g.espn_id AND l.source = 'lineup'
            WHERE (:since IS NULL OR g.kickoff >= :since)
              AND (:league IS NULL OR g.league = :league)
            ORDER BY g.kickoff, g.espn_id
        """
        rows = self.con.execute(sql, {"since": since, "league": league})
        return [dict(r) for r in rows]

    def game(self, espn_id: str) -> dict[str, Any] | None:
        """One game as in games(), plus its line-up; None when unknown."""
        match = [g for g in self.games() if g["espn_id"] == espn_id]
        if not match:
            return None
        rows = self.con.execute(
            "SELECT side, name, position, player_id, player, prev_min, form90 FROM lineups "
            "WHERE espn_id = ? ORDER BY side DESC, rowid",
            (espn_id,),
        )
        return {**match[0], "lineup": [dict(r) for r in rows]}
