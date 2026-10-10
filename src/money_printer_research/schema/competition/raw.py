"""Transfermarkt competitions.parquet, every column as exported (see schema/__init__)."""

import pandera.polars as pa
import polars as pl
from pandera.typing import FieldType as F


class TransfermarktCompetitionsRaw(pa.DataFrameModel):
    """<transfermarkt.out_dir>/competitions.parquet (export of the Transfermarkt DuckDB).

    Lookup table, one row per competition. Key: competition_id.
    """

    competition_id: F[str] = pa.Field(
        description=('Transfermarkt competition code: "GB1", "CL"; see TransfermarktGamesRaw.')
    )

    competition_code: F[str] = pa.Field(
        nullable=True, description=('URL slug: "premier-league", "uefa-champions-league".')
    )

    name: F[str] = pa.Field(
        nullable=True,
        description="Same slug as competition_code in this export, not a display name.",
    )

    sub_type: F[str] = pa.Field(
        nullable=True,
        description=(
            "Finer type: first_tier, domestic_cup, domestic_super_cup, play_off, or the "
            "competition itself for international ones (uefa_europa_league, world_cup)."
        ),
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

    country_id: F[pl.Int32] = pa.Field(
        nullable=True, description="Transfermarkt country id; -1 for international competitions."
    )

    country_name: F[str] = pa.Field(
        nullable=True, description=('Country: "England". Null for international competitions.')
    )

    domestic_league_code: F[str] = pa.Field(
        nullable=True,
        description=(
            'competition_id of the country\'s top league: "GB1" for the FA Cup. Null for '
            "international competitions."
        ),
    )

    confederation: F[str] = pa.Field(
        nullable=True,
        description=(
            'Region name as Transfermarkt writes it: "europa", "asien", "afrika", "amerika", '
            '"fifa".'
        ),
    )

    total_clubs: F[pl.Int32] = pa.Field(
        nullable=True,
        description=(
            "Clubs in the country's top league (20 for every English competition), not in "
            "this competition. Null for international competitions."
        ),
    )

    url: F[str] = pa.Field(nullable=True, description="Competition page on transfermarkt.co.uk.")


__all__ = ["TransfermarktCompetitionsRaw"]
