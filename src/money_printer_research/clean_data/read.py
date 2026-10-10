import polars as pl
from pandera.typing.polars import DataFrame

from ..config import Settings
from ..cool_stuff import returns
from ..schema import (
    TransfermarktCompetitionsRaw,
    TransfermarktGamesRaw,
    UnderstatScheduleRaw,
    UnderstatPlayerRaw,
)


@returns(UnderstatScheduleRaw)
def read_match_stat(settings: Settings) -> DataFrame[UnderstatScheduleRaw]:
    df = pl.read_csv(settings.understat.schedule_file, try_parse_dates=True)
    return UnderstatScheduleRaw.validate(df, lazy=True)


@returns(TransfermarktGamesRaw)
def read_match_info(settings: Settings) -> DataFrame[TransfermarktGamesRaw]:
    df = pl.read_parquet(settings.transfermarkt.out_dir / "games.parquet")
    return TransfermarktGamesRaw.validate(df, lazy=True)


@returns(TransfermarktCompetitionsRaw)
def read_league_info(settings: Settings) -> DataFrame[TransfermarktCompetitionsRaw]:
    df = pl.read_parquet(settings.transfermarkt.out_dir / "competitions.parquet")
    return TransfermarktCompetitionsRaw.validate(df, lazy=True)


@returns(UnderstatPlayerRaw)
def read_player_stat(settings: Settings) -> pl.DataFrame:
    files = sorted(settings.understat.out_dir.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.understat.out_dir}/<league>/")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
