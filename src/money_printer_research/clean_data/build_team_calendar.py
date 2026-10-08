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


def join_match_w_stat(
    tm: pl.DataFrame, clubs: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    """Understat schedule rows with their Transfermarkt league game (left join).

    Keyed on league, season and both teams rather than the date: a fixture is
    played once per season, and postponed or resumed games can sit days apart
    in the two sources (Udinese-Roma 2024: 14 April vs 25 April). Transfermarkt
    columns get a tm_ prefix so Understat's goals and date keep their names.
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
        .select(
            "league",
            "season_start",
            "home_team",
            "away_team",
            "tm_game_id",
            pl.col("season").alias("tm_season"),
            pl.col("round").alias("tm_round"),
            pl.col("home_formation"),
            pl.col("away_formation"),
            pl.col("home_goals").alias("tm_home_goals"),
            pl.col("away_goals").alias("tm_away_goals"),
        )
    )
    keys = ["league", "season_start", "home_team", "away_team"]
    return schedule.join(tm_league, on=keys, how="left", validate="1:1")


def game_info(tm: pl.DataFrame, game_map: pl.DataFrame) -> pl.DataFrame:
    """Per Understat game_id: Transfermarkt id, season, round, formations and official score."""
    return tm.join(game_map, on="tm_game_id", how="inner").select(
        "game_id",
        "tm_game_id",
        pl.col("season").alias("tm_season"),
        pl.col("round").alias("tm_round"),
        pl.col("home_club_formation").alias("home_formation"),
        pl.col("away_club_formation").alias("away_formation"),
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
