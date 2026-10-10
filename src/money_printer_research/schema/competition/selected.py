"""The columns of competitions.parquet the clean pipeline reads."""

import pandera.polars as pa
from pandera.typing import FieldType as F


class TransfermarktCompetitions(pa.DataFrameModel):
    """The Transfermarkt competition columns the clean pipeline reads."""

    competition_id: F[str] = pa.Field(
        description=('Transfermarkt competition code: "GB1", "CL"; see TransfermarktGamesRaw.')
    )

    type: F[str] = pa.Field(
        nullable=True,
        description=(
            "domestic_league, domestic_cup, international_cup, national_team_competition or "
            "other. Transfermarkt files the Europa League, Conference League and domestic "
            'super cups under "other"; only the Champions League, its qualifying and the UEFA'
            " Super Cup are international_cup."
        ),
    )


__all__ = ["TransfermarktCompetitions"]
