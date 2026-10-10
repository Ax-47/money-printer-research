"""clubs.csv: which Understat team each Transfermarkt club is."""

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F


class ClubMapCols(pa.DataFrameModel):
    """clubs.csv: Transfermarkt club_id -> Understat team, from map_club_w_team.

    Learned from league fixtures (same league, score and date), not from names, so
    it holds even where the two sources spell a club differently. Key: club_id.
    """

    club_id: F[pl.Int32] = pa.Field(
        unique=True,
        description=(
            "Transfermarkt club id (home_club_id / away_club_id in match_leagues.csv). Unique."
        ),
    )

    team: F[str] = pa.Field(
        description="Understat team name the club pairs with most often, after norm_team_names."
    )

    n: F[pl.UInt32] = pa.Field(
        description=(
            "League fixtures that back the pairing: Transfermarkt games whose league, score"
            " and date (within max_day_shift days) match an Understat game with this team "
            "on the same side."
        )
    )

    games: F[pl.UInt32] = pa.Field(
        description="All the club's Transfermarkt league games, home and away."
    )

    coverage: F[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description=(
            "n / games, rounded to 3 places. Close to 1.0 when the pairing is right; "
            "map_club_w_team rejects below 0.9 once games >= 10."
        ),
    )

    duplicate: F[bool] = pa.Field(
        description=(
            "True when another club_id pairs with the same team; map_club_w_team rejects "
            "any True row."
        )
    )


__all__ = ["ClubMapCols"]
