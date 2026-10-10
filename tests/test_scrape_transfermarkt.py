"""Transfermarkt download and export, against a tiny local DuckDB: no network."""

import io
import urllib.request
from pathlib import Path
from typing import Any, Self

import duckdb
import polars as pl
import pytest

from money_printer_research.config import Settings, TransfermarktSettings
from money_printer_research.data_collection import scrape_transfermarket as tm

# game_id, competition_id, season, home_club_id, away_club_id
GAMES = [
    (1, "GB1", "2020", 100, 101),  # in-scope league game
    (2, "FAC", "2020", 100, 200),  # cup game of an in-scope club: kept
    (3, "FAC", "2020", 300, 301),  # cup game with no in-scope club: dropped
    (4, "GB1", "2019", 100, 101),  # league game before first_season: dropped
    (5, "GB1", "2018", 100, 101),  # two seasons back: dropped everywhere
]
# appearance_id, game_id, player_id
APPEARANCES = [
    ("a1", 1, 7),  # player 7 played an in-scope game
    ("a2", 4, 7),  # one season before first_season: kept for rolling windows
    ("a3", 5, 7),  # two seasons before: dropped
    ("a4", 3, 8),  # player 8 never played in scope: dropped
]


def _make_db(path: Path) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE competitions (competition_id VARCHAR, name VARCHAR)")
    con.execute("INSERT INTO competitions VALUES ('GB1', 'premier-league'), ('FAC', 'fa-cup')")
    con.execute(
        "CREATE TABLE games (game_id INTEGER, competition_id VARCHAR, season VARCHAR,"
        " home_club_id INTEGER, away_club_id INTEGER)"
    )
    con.executemany("INSERT INTO games VALUES (?, ?, ?, ?, ?)", GAMES)
    con.execute("CREATE TABLE club_games (game_id INTEGER, club_id INTEGER)")
    con.executemany(
        "INSERT INTO club_games VALUES (?, ?)",
        [(g[0], club) for g in GAMES for club in (g[3], g[4])],
    )
    con.execute(
        "CREATE TABLE appearances (appearance_id VARCHAR, game_id INTEGER, player_id INTEGER)"
    )
    con.executemany("INSERT INTO appearances VALUES (?, ?, ?)", APPEARANCES)
    con.close()
    return path


@pytest.fixture
def db_file(tmp_path: Path) -> Path:
    return _make_db(tmp_path / "tm.duckdb")


@pytest.fixture
def no_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    """The toy tables have only a few columns, so skip the raw schema checks."""
    monkeypatch.setattr(tm, "SCHEMAS", {})


@pytest.mark.usefixtures("no_schemas")
def test_export_keeps_only_games_and_appearances_in_scope(db_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    tm.export(db_file, out, ["GB1"], 2020)

    games = pl.read_parquet(out / "games.parquet")
    assert sorted(games["game_id"].to_list()) == [1, 2]
    appearances = pl.read_parquet(out / "appearances.parquet")
    assert sorted(appearances["appearance_id"].to_list()) == ["a1", "a2"]
    competitions = pl.read_parquet(out / "competitions.parquet")
    assert competitions.height == 2
    assert not list(out.glob("*.part.parquet"))


def test_export_rejects_empty_league_list(db_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="league_ids is empty"):
        tm.export(db_file, tmp_path / "out", [], 2020)


def test_export_leaves_no_file_when_a_table_fails_its_schema(
    db_file: Path, tmp_path: Path
) -> None:
    # The real competitions schema needs columns the toy table lacks.
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        tm.export(db_file, out, ["GB1"], 2020)
    assert list(out.iterdir()) == []


class FakeResponse:
    """What urlopen returns: a context manager with headers and read()."""

    def __init__(self, body: bytes, length: bool = True, fail_after: int | None = None) -> None:
        self._body = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body))} if length else {}
        self._reads = 0
        self._fail_after = fail_after

    def __enter__(self) -> Self:
        """Open the response."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the response."""
        return None

    def read(self, size: int) -> bytes:
        """Next chunk of the body, or ConnectionResetError once past fail_after reads."""
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise ConnectionResetError("connection dropped")
        return self._body.read(size)


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, response: FakeResponse
) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request) -> FakeResponse:
        requests.append(request)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requests


def test_download_streams_to_dest_with_curl_user_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tm, "CHUNK_SIZE", 4)
    monkeypatch.setattr(tm, "PROGRESS_EVERY", 8)
    requests = _fake_urlopen(monkeypatch, FakeResponse(b"0123456789abcdef"))
    dest = tmp_path / "sub" / "db.duckdb"

    tm._download("https://example.invalid/db", dest)  # pyright: ignore[reportPrivateUsage]

    assert dest.read_bytes() == b"0123456789abcdef"
    assert not dest.with_name("db.duckdb.part").exists()
    assert requests[0].get_header("User-agent") == "curl/8.5.0"
    # Progress is printed once per PROGRESS_EVERY bytes.
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_download_never_leaves_a_partial_file_at_dest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tm, "CHUNK_SIZE", 4)
    _fake_urlopen(monkeypatch, FakeResponse(b"0123456789abcdef", length=False, fail_after=2))
    dest = tmp_path / "db.duckdb"

    with pytest.raises(ConnectionResetError):
        tm._download("https://example.invalid/db", dest)  # pyright: ignore[reportPrivateUsage]
    assert not dest.exists()


def test_print_progress_without_total(capsys: pytest.CaptureFixture[str]) -> None:
    tm._print_progress(3 * tm.MB, 0)  # pyright: ignore[reportPrivateUsage]
    tm._print_progress(3 * tm.MB, 10 * tm.MB)  # pyright: ignore[reportPrivateUsage]
    assert capsys.readouterr().out.splitlines() == ["  3 MB", "  3/10 MB"]


def _tm_settings(tmp_path: Path) -> Settings:
    return Settings(
        transfermarkt=TransfermarktSettings(
            url="https://example.invalid/db",
            db_file=tmp_path / "tm.duckdb",
            out_dir=tmp_path / "out",
            leagues={"GB1": "ENG-Premier League"},
            first_season=2020,
        )
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    """Record _download and export calls instead of running them."""
    recorded: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(tm, "_download", lambda *a: recorded.append(("download", a)))
    monkeypatch.setattr(tm, "export", lambda *a: recorded.append(("export", a)))
    return recorded


def test_collect_transfermarkt_downloads_once_then_exports(
    tmp_path: Path, calls: list[tuple[str, tuple[Any, ...]]]
) -> None:
    tm.collect_transfermarkt(_tm_settings(tmp_path))

    assert calls == [
        ("download", ("https://example.invalid/db", tmp_path / "tm.duckdb")),
        ("export", (tmp_path / "tm.duckdb", tmp_path / "out", ["GB1"], 2020)),
    ]


def test_collect_transfermarkt_reuses_downloaded_db(
    tmp_path: Path, calls: list[tuple[str, tuple[Any, ...]]]
) -> None:
    (tmp_path / "tm.duckdb").touch()
    tm.collect_transfermarkt(_tm_settings(tmp_path))
    assert [name for name, _ in calls] == ["export"]


def test_collect_transfermarkt_skips_when_every_table_exists(
    tmp_path: Path, calls: list[tuple[str, tuple[Any, ...]]]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for table in tm.TABLES:
        (out / f"{table}.parquet").touch()
    tm.collect_transfermarkt(_tm_settings(tmp_path))
    assert calls == []
