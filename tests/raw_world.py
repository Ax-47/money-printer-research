"""A small but complete set of raw files, as collect_data would leave them.

Six Premier League teams play a double round robin in each of SEASONS, plus one FA Cup
game per season. Understat (schedule CSV and one player parquet per season) and
Transfermarkt (games and competitions parquet) agree on every league fixture, so
clean_data runs end to end on it. One unplayed fixture per season sits after the last
played round, for predict.
"""

import itertools
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandera.polars as pa
import polars as pl

from money_printer_research.schema.competition.raw import TransfermarktCompetitionsRaw
from money_printer_research.schema.match.raw import TransfermarktGamesRaw
from money_printer_research.config import (
    CleanSettings,
    LiveSettings,
    ModelingSettings,
    Settings,
    TransfermarktSettings,
    UnderstatSettings,
)

LEAGUE = "ENG-Premier League"
TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham", "Liverpool", "Southampton"]
SEASONS = [2014, 2015, 2016, 2017]
STARTERS = ["GK", "DR", "DC", "DC", "DL", "MR", "MC", "MC", "ML", "FW", "FW"]
SUBS = 2


def _complete(df: pl.DataFrame, model: type[pa.DataFrameModel]) -> pl.DataFrame:
    """Add the schema's columns `df` lacks, all null, and cast to the schema's types."""
    columns = model.to_schema().columns
    missing = [pl.lit(None).alias(c) for c in columns if c not in df.columns]
    df = df.with_columns(missing)
    return df.select(pl.col(c).cast(col.dtype.type) for c, col in columns.items())


def _season_code(year: int) -> int:
    return (year % 100) * 100 + (year + 1) % 100


def _fixtures(year: int) -> list[tuple[datetime, int, int]]:
    """Every ordered pair once, one game every four days from early August.

    One game a day: the club mapping matches fixtures on league, score and date, so two
    same-day games with the same score would pair each club with both teams.
    """
    pairs = list(itertools.permutations(range(len(TEAMS)), 2))
    start = datetime(year, 8, 9, 15)
    return [(start + timedelta(days=4 * i), h, a) for i, (h, a) in enumerate(pairs)]


def settings_for(root: Path) -> Settings:
    return Settings(
        understat=UnderstatSettings(
            leagues=[LEAGUE],
            start_year=SEASONS[0],
            end_year=SEASONS[-1] + 1,
            out_dir=root / "raw" / "player_stats",
            schedule_file=root / "raw" / "xg_schedule.csv",
        ),
        transfermarkt=TransfermarktSettings(out_dir=root / "raw" / "transfermarkt"),
        clean=CleanSettings(out_dir=root / "clean"),
        modeling=ModelingSettings(
            out_file=root / "models" / "model.joblib",
            predictions_file=root / "predictions" / "upcoming.csv",
            odds_dir=root / "raw" / "odds",
            backtest_from=SEASONS[-1],
            backtest_file=root / "predictions" / "backtest.csv",
            learning_curve_file=root / "report" / "learning_curve.html",
        ),
        live=LiveSettings(db_file=root / "live.sqlite"),
    )


def write_raw(root: Path, seed: int = 0) -> Settings:
    """Write the raw files under `root` and return Settings pointing at them."""
    rng = np.random.default_rng(seed)
    settings = settings_for(root)
    schedule, games, odds = [], [], {}
    game_id = 1000
    for year in SEASONS:
        season_players = []
        fixtures = _fixtures(year)
        # One more, unplayed, a week after the last round.
        unplayed = (fixtures[-1][0] + timedelta(days=7), 0, 1)
        for n, (kickoff, h, a) in enumerate([*fixtures, unplayed]):
            played = n < len(fixtures)
            hg, ag = (int(rng.poisson(1.5)), int(rng.poisson(1.1))) if played else (None, None)
            home, away = TEAMS[h], TEAMS[a]
            game_id += 1
            schedule.append(
                {
                    "league": LEAGUE,
                    "season": _season_code(year),
                    "game": f"{kickoff:%Y-%m-%d} {home}-{away}",
                    "league_id": 1,
                    "season_id": year,
                    "game_id": game_id,
                    "date": kickoff,
                    "home_team_id": h + 1,
                    "away_team_id": a + 1,
                    "home_team": home,
                    "away_team": away,
                    "away_team_code": away[:3].upper(),
                    "home_team_code": home[:3].upper(),
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg": float(rng.uniform(0.3, 2.5)) if played else None,
                    "away_xg": float(rng.uniform(0.2, 2.0)) if played else None,
                    "is_result": played,
                    "has_data": played,
                    "url": f"https://understat.com/match/{game_id}",
                }
            )
            if not played:
                continue
            p = rng.dirichlet([5, 3, 3])
            odds.setdefault(year, []).append(
                {
                    "Div": "E0",
                    "Date": f"{kickoff:%d/%m/%Y}",
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": hg,
                    "FTAG": ag,
                    "HST": int(rng.integers(0, 8)),
                    "AST": int(rng.integers(0, 6)),
                    **{f"PS{k}": round(0.95 / q, 2) for k, q in zip("HDA", p, strict=True)},
                    **{f"PSC{k}": round(0.96 / q, 2) for k, q in zip("HDA", p, strict=True)},
                }
            )
            games.append(
                {
                    "game_id": f"tm{game_id}",
                    "competition_id": "GB1",
                    "season": str(year),
                    "round": f"{n // 2 + 1}. Matchday",
                    "date": kickoff.date(),
                    "home_club_id": 11 + h,
                    "away_club_id": 11 + a,
                    "home_club_goals": hg,
                    "away_club_goals": ag,
                    "home_club_formation": "4-4-2",
                    "away_club_formation": "4-2-3-1 Defensive" if n % 3 else "3-5-2",
                }
            )
            for side, team in ((0, h), (1, a)):
                for k, pos in enumerate(STARTERS + ["Sub"] * SUBS):
                    minutes = 90 if pos != "Sub" else int(rng.integers(5, 30))
                    season_players.append(
                        {
                            "league": LEAGUE,
                            "season": str(_season_code(year)),
                            "game": f"{kickoff:%Y-%m-%d} {home}-{away}",
                            "team": TEAMS[team],
                            "player": f"{TEAMS[team]} Player {k + 1}",
                            "league_id": "1",
                            "season_id": year,
                            "game_id": game_id,
                            "team_id": team + 1,
                            "player_id": 100 * (team + 1) + k,
                            "position": pos,
                            "position_id": k,
                            "minutes": minutes,
                            "goals": 0,
                            "own_goals": 0,
                            "shots": int(rng.integers(0, 3)),
                            "xg": float(rng.uniform(0, 0.4)),
                            "xg_chain": 0.5,
                            "xg_buildup": 0.2,
                            "assists": 0,
                            "xa": float(rng.uniform(0, 0.3)),
                            "key_passes": int(rng.integers(0, 3)),
                            "yellow_cards": 0,
                            "red_cards": 0,
                        }
                    )
        # A midweek FA Cup game, in no Understat schedule.
        cup_day = fixtures[2][0].date() + timedelta(days=3)
        games.append(
            {
                "game_id": f"cup{year}",
                "competition_id": "FAC",
                "season": str(year),
                "round": "Third Round",
                "date": cup_day,
                "home_club_id": 11,
                "away_club_id": 12,
                "home_club_goals": 1,
                "away_club_goals": 0,
                "home_club_formation": None,
                "away_club_formation": "4-4-2",
            }
        )
        path = settings.understat.out_dir / LEAGUE / f"{year}-{str(year + 1)[-2:]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(season_players).write_parquet(path)

    for year, rows in odds.items():
        path = settings.modeling.odds_dir / LEAGUE / f"{year}-{str(year + 1)[-2:]}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(path)
    settings.understat.schedule_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schedule).write_csv(settings.understat.schedule_file)
    tm = settings.transfermarkt.out_dir
    tm.mkdir(parents=True, exist_ok=True)
    _complete(pl.DataFrame(games), TransfermarktGamesRaw).write_parquet(tm / "games.parquet")
    competitions = pl.DataFrame(
        {"competition_id": ["GB1", "FAC"], "type": ["domestic_league", "domestic_cup"]}
    )
    _complete(competitions, TransfermarktCompetitionsRaw).write_parquet(
        tm / "competitions.parquet"
    )
    return settings


def last_kickoff() -> date:
    return _fixtures(SEASONS[-1])[-1][0].date()
