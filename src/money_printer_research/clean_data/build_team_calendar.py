"""Plan A: every club's full match calendar (league, cups, Europe) from Transfermarkt.

Understat only has league games, so a team that played in Europe on Wednesday
looks rested for Saturday. Transfermarkt `games` has all competitions. This
module:

1. maps Transfermarkt clubs to Understat team names from fixtures that appear
   in both sources (same league, score and date within a day), not from spelling;
2. maps Transfermarkt league games to Understat game_id;
3. computes pre-match schedule features on the full calendar and attaches them
   to each Understat game as h_* / a_* columns.

All features use only games before the match (and, for the look-ahead ones,
fixtures already scheduled, which are public before kick-off).
"""

import polars as pl

# Transfermarkt competition_id -> Understat league
TM_LEAGUE = {
    "GB1": "ENG-Premier League",
    "ES1": "ESP-La Liga",
    "L1": "GER-Bundesliga",
    "IT1": "ITA-Serie A",
    "FR1": "FRA-Ligue 1",
}

SCHEDULE_FEATURES = [
    "rest_days_all",  # days since the previous game in any competition
    "games_7d_all",  # games in the previous 7 days, any competition
    "games_14d_all",
    "non_league_14d",  # cup / European / other games in the previous 14 days
    "prev_europe_away",  # previous game was an away European tie (travel)
    "days_to_next_all",  # days until the next game in any competition
    "next_is_europe",  # the next game is European (rotation risk)
]


def standardize_games(games: pl.DataFrame, competitions: pl.DataFrame) -> pl.DataFrame:
    """Transfermarkt games with a comp_type column: league / domestic_cup / europe / other."""
    comp = competitions.select("competition_id", pl.col("type").alias("tm_type"))
    return (
        games.join(comp, on="competition_id", how="left")
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.when(pl.col("competition_id").is_in(list(TM_LEAGUE)))
            .then(pl.lit("league"))
            .when(pl.col("tm_type") == "domestic_cup")
            .then(pl.lit("domestic_cup"))
            .when(pl.col("tm_type") == "international_cup")
            .then(pl.lit("europe"))
            .otherwise(pl.lit("other"))
            .alias("comp_type"),
            pl.col("competition_id").replace_strict(TM_LEAGUE, default=None).alias("league"),
        )
        .select(
            pl.col("game_id").alias("tm_game_id"),
            "competition_id",
            "comp_type",
            "league",
            "date",
            "home_club_id",
            "away_club_id",
            pl.col("home_club_goals").cast(pl.Int64).alias("home_goals"),
            pl.col("away_club_goals").cast(pl.Int64).alias("away_goals"),
            "season",
            pl.col("season").cast(pl.Int32).alias("season_start"),
            "round",
            "home_club_formation",
            "away_club_formation",
        )
        .filter(pl.col("date").is_not_null())
    )


def map_clubs(tm: pl.DataFrame, schedule: pl.DataFrame, max_day_shift: int = 1) -> pl.DataFrame:
    """Transfermarkt club_id -> Understat team, learned from league fixtures.

    coverage: share of the club's Transfermarkt league games that back the pairing
              (close to 1.0 when correct). duplicate: two club ids map to one team.
    """
    tm_league = tm.filter(pl.col("comp_type") == "league")
    us = schedule.select(
        "league",
        pl.col("date").cast(pl.Date).alias("us_date"),
        "home_team",
        "away_team",
        pl.col("home_goals").cast(pl.Int64),
        pl.col("away_goals").cast(pl.Int64),
    )
    pairs = tm_league.join(us, on=["league", "home_goals", "away_goals"]).filter(
        (pl.col("date") - pl.col("us_date")).dt.total_days().abs() <= max_day_shift
    )
    long = pl.concat(
        [
            pairs.select(
                pl.col("home_club_id").alias("club_id"), pl.col("home_team").alias("team")
            ),
            pairs.select(
                pl.col("away_club_id").alias("club_id"), pl.col("away_team").alias("team")
            ),
        ]
    )
    games = (
        pl.concat(
            [
                tm_league.select(pl.col("home_club_id").alias("club_id")),
                tm_league.select(pl.col("away_club_id").alias("club_id")),
            ]
        )
        .group_by("club_id")
        .len("games")
    )
    return (
        long.group_by("club_id", "team")
        .len()
        .sort("len", descending=True)
        .group_by("club_id", maintain_order=True)
        .agg(pl.col("team").first(), pl.col("len").first().alias("n"))
        .join(games, on="club_id")
        .with_columns((pl.col("n") / pl.col("games")).round(3).alias("coverage"))
        .with_columns(pl.len().over("team").gt(1).alias("duplicate"))
        .sort("team")
    )


def map_games(tm: pl.DataFrame, clubs: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Transfermarkt league tm_game_id <-> Understat game_id (one to one).

    Keyed on league, season and both teams rather than the date: a fixture is
    played once per season, and postponed or resumed games can sit days apart
    in the two sources (Udinese-Roma 2024: 14 April vs 25 April).
    """
    club_team = clubs.filter(~pl.col("duplicate")).select("club_id", "team")
    tm_league = (
        tm.filter(pl.col("comp_type") == "league")
        .join(
            club_team.rename({"club_id": "home_club_id", "team": "home_team"}), on="home_club_id"
        )
        .join(
            club_team.rename({"club_id": "away_club_id", "team": "away_team"}), on="away_club_id"
        )
    )
    us = schedule.select("league", "season_start", "home_team", "away_team", "game_id")
    keys = ["league", "season_start", "home_team", "away_team"]
    return tm_league.join(us, on=keys, how="inner", validate="1:1").select("tm_game_id", "game_id")


def game_info(tm: pl.DataFrame, game_map: pl.DataFrame) -> pl.DataFrame:
    """Per Understat game_id: Transfermarkt id, season, round, formations and official score."""
    return tm.join(game_map, on="tm_game_id", how="inner").select(
        "game_id",
        "tm_game_id",
        pl.col("season").alias("tm_season"),
        pl.col("round").alias("tm_round"),
        pl.col("home_club_formation").alias("h_formation"),
        pl.col("away_club_formation").alias("a_formation"),
        pl.col("home_goals").alias("tm_home_goals"),
        pl.col("away_goals").alias("tm_away_goals"),
    )


def club_calendar(tm: pl.DataFrame) -> pl.DataFrame:
    """One row per club per game with the schedule features, computed on all competitions."""
    sides = [("home_club_id", "away_club_id", True), ("away_club_id", "home_club_id", False)]
    cal = pl.concat(
        [
            tm.select(
                "tm_game_id",
                "date",
                "comp_type",
                pl.col(own).alias("club_id"),
                pl.lit(is_home).alias("is_home"),
            )
            for own, _, is_home in sides
        ]
    ).sort("club_id", "date", "tm_game_id")

    def prior_count(days: int, cond: pl.Expr | None = None) -> pl.Expr:
        flag = pl.lit(1, dtype=pl.Int64) if cond is None else cond.cast(pl.Int64)
        return (
            pl.repeat(1, pl.len(), dtype=pl.Int64)
            .mul(flag)
            .rolling_sum_by("date", window_size=f"{days}d", closed="left")
            .over("club_id")
            .fill_null(0)
        )

    europe_away = (pl.col("comp_type") == "europe") & ~pl.col("is_home")
    return cal.with_columns(
        pl.col("date").diff().dt.total_days().over("club_id").alias("rest_days_all"),
        prior_count(7).alias("games_7d_all"),
        prior_count(14).alias("games_14d_all"),
        prior_count(14, pl.col("comp_type") != "league").alias("non_league_14d"),
        europe_away.shift(1).over("club_id").fill_null(False).alias("prev_europe_away"),
        (pl.col("date").shift(-1).over("club_id") - pl.col("date"))
        .dt.total_days()
        .alias("days_to_next_all"),
        (pl.col("comp_type").shift(-1).over("club_id") == "europe")
        .fill_null(False)
        .alias("next_is_europe"),
    )


def team_schedule_features(
    tm: pl.DataFrame, clubs: pl.DataFrame, game_map: pl.DataFrame
) -> pl.DataFrame:
    """Per Understat game_id: h_* and a_* schedule features from the full calendar."""
    cal = club_calendar(tm).join(game_map, on="tm_game_id", how="inner")
    keep = ["game_id", "is_home", *SCHEDULE_FEATURES]
    home = cal.filter(pl.col("is_home")).select(keep).drop("is_home")
    away = cal.filter(~pl.col("is_home")).select(keep).drop("is_home")
    return home.rename({c: f"h_{c}" for c in SCHEDULE_FEATURES}).join(
        away.rename({c: f"a_{c}" for c in SCHEDULE_FEATURES}),
        on="game_id",
        how="full",
        coalesce=True,
    )
