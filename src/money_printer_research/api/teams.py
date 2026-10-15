"""Per-team numbers for the API, from cleaned_matches.csv and elo.csv.

league_table  one league season: played, W/D/L, goals, xG per game, points, rank
team_stats    one team's current season: its table row, home and away splits,
              last games and its Elo before each of its last matches
"""

from typing import Any

import polars as pl

LAST_GAMES = 5
ELO_HISTORY = 10


def _team_rows(matches: pl.DataFrame) -> pl.DataFrame:
    """One row per team per played game, from the team's side."""
    common = ["game_id", "date", "league", "season_start"]
    sides = []
    for team, opp, venue in (("home", "away", "home"), ("away", "home", "away")):
        sides.append(
            matches.select(
                *common,
                pl.col(f"{team}_team").alias("team"),
                pl.col(f"{opp}_team").alias("opponent"),
                pl.lit(venue).alias("venue"),
                pl.col(f"{team}_goals").alias("gf"),
                pl.col(f"{opp}_goals").alias("ga"),
                pl.col(f"{team}_xg").alias("xg_for"),
                pl.col(f"{opp}_xg").alias("xg_against"),
            )
        )
    rows = pl.concat(sides)
    return rows.with_columns(
        pl.when(pl.col("gf") > pl.col("ga"))
        .then(pl.lit("W"))
        .when(pl.col("gf") == pl.col("ga"))
        .then(pl.lit("D"))
        .otherwise(pl.lit("L"))
        .alias("result")
    )


def _summary(rows: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        rows.group_by(by)
        .agg(
            pl.len().alias("played"),
            (pl.col("result") == "W").sum().alias("won"),
            (pl.col("result") == "D").sum().alias("drawn"),
            (pl.col("result") == "L").sum().alias("lost"),
            pl.col("gf").sum().alias("goals_for"),
            pl.col("ga").sum().alias("goals_against"),
            pl.col("xg_for").mean().round(2).alias("xg_for_per_game"),
            pl.col("xg_against").mean().round(2).alias("xg_against_per_game"),
        )
        .with_columns(
            (pl.col("goals_for") - pl.col("goals_against")).alias("goal_diff"),
            (3 * pl.col("won") + pl.col("drawn")).alias("points"),
        )
    )


def latest_season(matches: pl.DataFrame, league: str) -> int | None:
    seasons = matches.filter(pl.col("league") == league)["season_start"]
    return int(seasons.max()) if seasons.len() else None  # pyright: ignore[reportArgumentType]


def league_table(matches: pl.DataFrame, league: str, season: int) -> pl.DataFrame:
    """The table of one league season, best first: points, goal difference, goals."""
    games = matches.filter((pl.col("league") == league) & (pl.col("season_start") == season))
    table = _summary(_team_rows(games), ["team"]).sort(
        ["points", "goal_diff", "goals_for", "team"], descending=[True, True, True, False]
    )
    return table.with_row_index("rank", offset=1)


def teams(matches: pl.DataFrame) -> pl.DataFrame:
    """Every team with its league and latest season."""
    rows = _team_rows(matches)
    return (
        rows.sort("date")
        .group_by("team")
        .agg(pl.col("league").last(), pl.col("season_start").last())
        .sort("league", "team")
    )


def team_stats(matches: pl.DataFrame, elo: pl.DataFrame, team: str) -> dict[str, Any] | None:
    """`team`'s current season, or None when it has no played game."""
    rows = _team_rows(matches).filter(pl.col("team") == team)
    if rows.is_empty():
        return None
    last = rows.sort("date").row(-1, named=True)
    league, season = last["league"], last["season_start"]
    season_rows = rows.filter(pl.col("season_start") == season)
    table = league_table(matches, league, season)
    elo_rows = (
        elo.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
        .sort("match_date")
        .tail(ELO_HISTORY)
        .select(
            "match_date",
            pl.when(pl.col("home_team") == team)
            .then(pl.col("elo_home"))
            .otherwise(pl.col("elo_away"))
            .round(1)
            .alias("elo"),
        )
    )
    recent = season_rows.sort("date", descending=True).head(LAST_GAMES)
    return {
        "team": team,
        "league": league,
        "season_start": season,
        "table": table.filter(pl.col("team") == team).drop("team").to_dicts()[0],
        "by_venue": _summary(season_rows, ["venue"]).sort("venue", descending=True).to_dicts(),
        "form": "".join(recent["result"].to_list()),
        "last_games": recent.select(
            "date", "opponent", "venue", "gf", "ga", "result", "xg_for", "xg_against"
        ).to_dicts(),
        "elo_history": elo_rows.to_dicts(),
    }
