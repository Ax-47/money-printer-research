from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "settings.toml").exists():
            return p
    raise FileNotFoundError(f"settings.toml not found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)


def _from_root(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


RootPath = Annotated[Path, AfterValidator(_from_root)]


class _Section(BaseModel):
    # Run validators on defaults too, so default paths also resolve against ROOT.
    model_config = ConfigDict(validate_default=True)


class KaggleSettings(_Section):
    handle: str = "marcohuiii/english-premier-league-epl-match-data-2000-2025"
    output_dir: RootPath = Path("data")


class FBrefSettings(_Section):
    league: str = "ENG-Premier League"
    start_year: int = 2015  # first season starts in this year
    end_year: int = 2025  # exclusive: last season is 2024-25
    out_dir: RootPath = Path("data/lineups")
    cache_dir: RootPath = Path("data/FBref")


class UnderstatSettings(_Section):
    league: str = "ENG-Premier League"
    start_year: int = 2014  # first season starts in this year
    end_year: int = 2025  # exclusive: last season is 2024-25
    out_dir: RootPath = Path("data/player_stats")
    schedule_file: RootPath = Path("data/xg_schedule.csv")
    cache_dir: RootPath = Path("data/Understat")


class CleanSettings(_Section):
    out_dir: RootPath = Path("data/clean")


class TeamMapSettings(_Section):
    out_file: RootPath = Path("data/team_map.csv")


class Settings(BaseSettings):
    """Main settings. Priority: init args > env vars > settings.toml > defaults."""

    model_config = SettingsConfigDict(
        toml_file=ROOT / "settings.toml",
        env_nested_delimiter="__",  # e.g. FBREF__START_YEAR=2018
        extra="ignore",
    )

    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)
    fbref: FBrefSettings = Field(default_factory=FBrefSettings)
    understat: UnderstatSettings = Field(default_factory=UnderstatSettings)
    team_map: TeamMapSettings = Field(default_factory=TeamMapSettings)
    clean: CleanSettings = Field(default_factory=CleanSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init args, then env vars, then settings.toml."""
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))


settings = Settings()
