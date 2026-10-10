from dataclasses import dataclass

import polars as pl
from pandera.typing.polars import DataFrame

from ..config import Settings
from ..config import settings as default_settings
from ..cool_stuff import Err, Ok, Result, all_ok, check, monad
from ..exact_features import (
    get_elo,
    get_fatigue,
    get_player_features,
    team_schedule_features,
    to_elo_input,
)
from ..schema import (
    CleanedMatchCols,
    PlayerStatCols,
    ClubMapCols,
    MatchFeaturesCols,
    MatchLeaguesCols,
    EloCols,
    TeamFatigueCols,
    TeamScheduleCols,
)
from ..schema.competition.selected import TransfermarktCompetitions
from ..schema.match.selected import TransfermarktGames
from ..utils import SavesFrames
from .formation import encode_formation
from .join import join_features
from .map_club_team import map_club_w_team
from .match import join_match_w_stat
from .match_leagues import join_match_w_league
from .match_stat import clean_match_stat
from .player_stat import clean_player_stat
from .read import read_league_info, read_match_info, read_match_stat, read_player_stat

MAX_UNMATCHED_SHARE = 0.01


@dataclass(frozen=True)
class Cleaned(SavesFrames):
    """The clean outputs, one per CSV in settings.clean.out_dir (<field name>.csv)."""

    match_leagues: DataFrame[MatchLeaguesCols]
    clubs: DataFrame[ClubMapCols]
    cleaned_matches: DataFrame[CleanedMatchCols]
    player_stat: DataFrame[PlayerStatCols]


@dataclass(frozen=True)
class ExactFeatures(SavesFrames):
    """Pre-match features, and the matches with all of them joined on."""

    elo: DataFrame[EloCols]
    fatigue: DataFrame[TeamFatigueCols]
    schedule: DataFrame[TeamScheduleCols]
    players: pl.DataFrame
    match_features: DataFrame[MatchFeaturesCols]


@monad
def clean(settings: Settings) -> Result[Cleaned]:
    """Run every step; Ok(Cleaned) or the first failing step's Err."""
    cleaned_match_stat = read_match_stat(settings).and_then(clean_match_stat).if_err()
    player_stat = read_player_stat(settings).and_then(clean_player_stat).if_err()
    match_leagues = (
        all_ok(
            read_match_info(settings) >> check(TransfermarktGames) >> encode_formation,
            read_league_info(settings) >> check(TransfermarktCompetitions),
        )
        .and_then(lambda pair: join_match_w_league(*pair))
        .if_err()
    )
    clubs = map_club_w_team(match_leagues, cleaned_match_stat).if_err()
    cleaned_matches = join_match_w_stat(match_leagues, clubs, cleaned_match_stat).if_err()

    unmatched = cleaned_matches[CleanedMatchCols.tm_game_id].null_count()
    if unmatched > MAX_UNMATCHED_SHARE * cleaned_matches.height:
        return Err(
            f"join_match_w_stat: {unmatched:,} of {cleaned_matches.height:,} matches "
            "have no Transfermarkt game"
        )

    return Ok(
        Cleaned(
            match_leagues=match_leagues,
            clubs=clubs,
            cleaned_matches=cleaned_matches,
            player_stat=player_stat,
        )
    )


@monad
def exact_features(cleaned: Cleaned) -> Result[ExactFeatures]:
    """Elo, fatigue, schedule and player-slot features, joined onto the matches.

    The four features are computed independently, so every failure among them comes
    back in one Err.
    """
    match_leagues, cleaned_matches, player_stat = (
        cleaned.match_leagues,
        cleaned.cleaned_matches,
        cleaned.player_stat,
    )
    elo, fatigue, schedule, players = all_ok(
        to_elo_input(cleaned_matches) >> get_elo,
        get_fatigue(player_stat, cleaned_matches),
        team_schedule_features(match_leagues, cleaned_matches),
        get_player_features(player_stat, cleaned_matches),
    ).if_err()
    match_features = join_features(cleaned_matches, elo, fatigue, schedule, players).if_err()
    return Ok(
        ExactFeatures(
            elo=elo,
            fatigue=fatigue,
            schedule=schedule,
            players=players,
            match_features=match_features,
        )
    )


def clean_data(settings: Settings = default_settings) -> tuple[Cleaned, ExactFeatures]:
    """Entry point `clean_data`: clean, then exact features; raise on Err, write both as CSV.

    The clean outputs are written before the features run, so a feature failure
    leaves them on disk.
    """
    out_dir = settings.clean.out_dir
    cleaned = clean(settings).expect("clean")
    for path in cleaned.save(out_dir):
        print("saved", path, flush=True)

    features = exact_features(cleaned).expect("exact_features")
    for path in features.save(out_dir):
        print("saved", path, flush=True)
    return cleaned, features
