import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.config import Settings
from money_printer_research.cool_stuff import returns
from money_printer_research.schema.raw import LEAGUE_INFO, MATCH_INFO, MATCH_STAT, PLAYER_STAT


@returns(MATCH_STAT)
def read_match_stat(settings: Settings) -> DataFrame[MATCH_STAT]:
    df = pl.read_csv(settings.understat.schedule_file, try_parse_dates=True)
    return MATCH_STAT.validate(df, lazy=True)


@returns(MATCH_INFO)
def read_match_info(settings: Settings) -> DataFrame[MATCH_INFO]:
    df = pl.read_parquet(settings.transfermarkt.out_dir / "games.parquet")
    return MATCH_INFO.validate(df, lazy=True)


@returns(LEAGUE_INFO)
def read_league_info(settings: Settings) -> DataFrame[LEAGUE_INFO]:
    df = pl.read_parquet(settings.transfermarkt.out_dir / "competitions.parquet")
    return LEAGUE_INFO.validate(df, lazy=True)


@returns(PLAYER_STAT)
def read_player_stat(settings: Settings) -> pl.DataFrame:
    files = sorted(settings.understat.out_dir.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.understat.out_dir}/<league>/")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
