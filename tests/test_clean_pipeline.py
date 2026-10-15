"""clean_data end to end on tests/raw_world.py's raw files, and its failure paths."""

from pathlib import Path

import polars as pl
import pytest
from pandera.typing.polars import DataFrame

from money_printer_research.clean_data import clean, clean_data
from money_printer_research.clean_data.formation import unknown_formations
from money_printer_research.clean_data.map_club_team import map_club_w_team
from money_printer_research.clean_data.read import read_match_stat, read_player_stat
from money_printer_research.clean_data.rename_team_names import alias_block
from money_printer_research.config import Settings, UnderstatSettings
from money_printer_research.schema import CLEAN_FILES, MatchLeaguesCols, read_clean
from money_printer_research.schema.match.raw import UnderstatScheduleRaw
from tests.raw_world import SEASONS, STARTERS, TEAMS, write_raw

PLAYED_PER_SEASON = len(TEAMS) * (len(TEAMS) - 1)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = write_raw(tmp_path_factory.mktemp("world"))
    clean_data(settings)
    return settings


def test_every_clean_file_is_written_and_valid(world: Settings) -> None:
    for name in CLEAN_FILES:
        assert read_clean(world.clean.out_dir, name).height > 0, name


def test_matches_join_transfermarkt_and_keep_every_played_game(world: Settings) -> None:
    matches = read_clean(world.clean.out_dir, "cleaned_matches")
    assert matches.height == PLAYED_PER_SEASON * len(SEASONS)
    assert matches["tm_game_id"].null_count() == 0
    assert set(matches["home_team"]) == {t.lower() for t in TEAMS}
    assert set(matches["home_formation"]) == {"4-4-2"}


def test_clubs_map_one_to_one(world: Settings) -> None:
    clubs = read_clean(world.clean.out_dir, "clubs")
    assert dict(zip(clubs["club_id"], clubs["team"], strict=True)) == {
        11 + i: t.lower() for i, t in enumerate(TEAMS)
    }
    assert clubs["coverage"].to_list() == [1.0] * len(TEAMS)


def test_competition_types(world: Settings) -> None:
    leagues = read_clean(world.clean.out_dir, "match_leagues")
    counts = dict(leagues.group_by("comp_type").len().iter_rows())
    assert counts == {"league": PLAYED_PER_SEASON * len(SEASONS), "domestic_cup": len(SEASONS)}


def test_player_slots_hold_eleven_starters(world: Settings) -> None:
    players = read_clean(world.clean.out_dir, "players")
    last = players.sort("game_id").row(-1, named=True)
    # By the last game every starter has played enough to have per-90 rates.
    assert all(last[f"h_player{i}_xg90"] is not None for i in range(1, len(STARTERS) + 1))
    assert players.height == PLAYED_PER_SEASON * len(SEASONS)


def test_unplayed_fixtures_are_kept_out_of_the_clean_matches(world: Settings) -> None:
    raw = read_match_stat(world).unwrap()
    assert raw.filter(~pl.col("is_result")).height == len(SEASONS)


def test_read_player_stat_needs_files(tmp_path: Path) -> None:
    settings = Settings(understat=UnderstatSettings(out_dir=tmp_path))
    result = read_player_stat(settings)
    assert result.is_err() and "No parquet files" in str(result.reason)  # pyright: ignore[reportAttributeAccessIssue]


def test_clean_fails_when_too_many_matches_miss_transfermarkt(tmp_path: Path) -> None:
    settings = write_raw(tmp_path)
    games = settings.transfermarkt.out_dir / "games.parquet"
    tm = pl.read_parquet(games)
    # Drop one season's league games: its Understat matches find no Transfermarkt game.
    tm.filter(~((pl.col("season") == "2015") & (pl.col("competition_id") == "GB1"))).write_parquet(
        games
    )
    result = clean(settings)
    assert result.is_err()
    assert "have no Transfermarkt game" in str(result.reason)  # pyright: ignore[reportAttributeAccessIssue]


def test_clean_reports_matches_after_transfermarkt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = write_raw(tmp_path)
    games = settings.transfermarkt.out_dir / "games.parquet"
    pl.read_parquet(games).filter(pl.col("season") != str(SEASONS[-1])).write_parquet(games)
    assert clean(settings).is_ok()
    assert "after Transfermarkt's last season" in capsys.readouterr().out


def _tables(
    world: Settings,
) -> tuple[DataFrame[MatchLeaguesCols], DataFrame[UnderstatScheduleRaw]]:
    leagues = DataFrame[MatchLeaguesCols](read_clean(world.clean.out_dir, "match_leagues"))
    return leagues, read_match_stat(world).unwrap()


def _typed(df: pl.DataFrame) -> DataFrame[MatchLeaguesCols]:
    return DataFrame[MatchLeaguesCols](df)


def test_map_club_w_team_errors(world: Settings) -> None:
    from money_printer_research.clean_data.match_stat import clean_match_stat

    leagues, raw = _tables(world)
    schedule = clean_match_stat(raw).unwrap()

    no_league = map_club_w_team(_typed(leagues.filter(pl.col("comp_type") != "league")), schedule)
    assert "no league games" in str(no_league.reason)  # pyright: ignore[reportAttributeAccessIssue]

    shifted = leagues.with_columns(pl.col("date") + pl.duration(days=30))
    unmatched = map_club_w_team(_typed(shifted), schedule)
    assert "no club matched" in str(unmatched.reason)  # pyright: ignore[reportAttributeAccessIssue]

    # Two club ids for one team: the second club's games point at the first's team.
    twin = leagues.with_columns(
        pl.when(pl.col("home_club_id") == 12)
        .then(99)
        .otherwise(pl.col("home_club_id"))
        .alias("home_club_id")
    )
    doubtful = map_club_w_team(_typed(twin), schedule)
    assert "doubtful" in str(doubtful.reason)  # pyright: ignore[reportAttributeAccessIssue]


def test_unknown_formations_lists_shapes_that_do_not_add_up() -> None:
    games = pl.DataFrame(
        {
            "home_club_formation": ["4-4-2", "4-4-3", "diamond", None],
            "away_club_formation": ["3-5-2", " 4-4-2 ", "4-4-2", "4-4-2"],
        }
    )
    assert unknown_formations(games)["formation"].to_list() == ["4-4-3", "diamond"]  # pyright: ignore[reportArgumentType]


def test_alias_block_lists_unknown_teams_by_league() -> None:
    schedule = pl.DataFrame(
        {
            "league": ["ENG-Premier League", "ENG-Premier League", "ESP-La Liga"],
            "home_team": ["Arsenal", "Nowhere Rovers", "Atletico Somewhere"],
            "away_team": ["Chelsea", "Arsenal", "Arsenal"],
        }
    )
    assert alias_block(schedule) == (
        "    # ENG-Premier League\n"
        '    "nowhere rovers": "nowhere rovers",\n'
        "    # ESP-La Liga\n"
        '    "atletico somewhere": "atletico somewhere",'
    )
