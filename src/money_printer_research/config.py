"""Typed project settings, validated when they are loaded.

Values come from, highest priority first:

1. arguments passed to `Settings(...)` (tests, notebooks)
2. environment variables, sections joined with `__`: `UNDERSTAT__START_YEAR=2018`
3. a `.env` file in the working directory, same names as the environment variables
4. `settings.toml`
5. the defaults in this module

`settings.toml` is the file named by `MPR_SETTINGS_FILE`, else the first one found
walking up from the working directory, else from this package. Relative paths in it
resolve against the folder that holds it (the project root), not the working directory.

Run `uv run show_config` to print the settings in effect and where they came from.
"""

import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

SETTINGS_FILE_ENV = "MPR_SETTINGS_FILE"
SETTINGS_FILE_NAME = "settings.toml"


class ConfigError(ValueError):
    """The settings could not be loaded: a missing file or an invalid value."""


def find_settings_file(env: dict[str, str] | None = None, cwd: Path | None = None) -> Path | None:
    """The settings.toml to load, or None when there is none (defaults and env only)."""
    env = dict(os.environ) if env is None else env
    if explicit := env.get(SETTINGS_FILE_ENV):
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"{SETTINGS_FILE_ENV}={explicit}: no such file ({path})")
        return path
    starts = [(cwd or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for start in starts:
        for folder in [start, *start.parents]:
            if (folder / SETTINGS_FILE_NAME).is_file():
                return folder / SETTINGS_FILE_NAME
    return None


SETTINGS_FILE = find_settings_file()
# Relative paths resolve against the project root: the folder holding settings.toml.
ROOT = SETTINGS_FILE.parent if SETTINGS_FILE else Path.cwd().resolve()


def _from_root(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


RootPath = Annotated[Path, AfterValidator(_from_root)]

# Transfermarkt competition ids go into SQL, so only letters and digits: "GB1".
_TM_ID = re.compile(r"[A-Z0-9]+")
FIRST_UNDERSTAT_SEASON = 2014


class _Section(BaseModel):
    # Run validators on defaults too, so default paths also resolve against ROOT.
    # Unknown keys are errors, so a typo in settings.toml is not silently ignored.
    model_config = ConfigDict(validate_default=True, extra="forbid")


def _no_duplicates[T](items: list[T]) -> list[T]:
    if dupes := sorted({str(x) for x in items if items.count(x) > 1}):
        raise ValueError(f"listed more than once: {', '.join(dupes)}")
    return items


class TransfermarktSettings(_Section):
    url: str = Field(
        default="https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb",
        description="Where to download the transfermarkt-datasets DuckDB file.",
    )
    db_file: RootPath = Field(
        default=Path("data/Transfermarkt/transfermarkt-datasets.duckdb"),
        description="Local copy of the DuckDB file; downloaded only when missing.",
    )
    out_dir: RootPath = Field(
        default=Path("data/raw/transfermarkt"), description="Exported parquet tables."
    )
    leagues: dict[str, str] = Field(
        default={
            "GB1": "ENG-Premier League",
            "ES1": "ESP-La Liga",
            "L1": "GER-Bundesliga",
            "IT1": "ITA-Serie A",
            "FR1": "FRA-Ligue 1",
        },
        min_length=1,
        description="Transfermarkt competition id -> Understat league name.",
    )
    first_season: int = Field(
        default=2014, ge=1990, description="First season (start year) to export."
    )

    @model_validator(mode="after")
    def _check_league_ids(self) -> Self:
        if bad := [k for k in self.leagues if not _TM_ID.fullmatch(k)]:
            raise ValueError(
                f"leagues: {', '.join(map(repr, bad))} is not a Transfermarkt competition id"
                ' (upper-case letters and digits, like "GB1")'
            )
        return self


class UnderstatSettings(_Section):
    leagues: Annotated[list[str], AfterValidator(_no_duplicates)] = Field(
        default=[
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ],
        min_length=1,
        description="soccerdata league names to scrape.",
    )
    start_year: int = Field(
        default=2014,
        ge=FIRST_UNDERSTAT_SEASON,
        description="Start year of the first season (Understat begins in 2014).",
    )
    end_year: int = Field(
        default=2026, description="Exclusive: 2026 makes 2025-26 the last season."
    )
    out_dir: RootPath = Field(
        default=Path("data/raw/player_stats"), description="Player stats, one parquet per match."
    )
    schedule_file: RootPath = Field(
        default=Path("data/raw/xg_schedule.csv"), description="Schedule with goals and xG."
    )
    cache_dir: RootPath = Field(default=Path("data/Understat"), description="soccerdata cache.")

    @model_validator(mode="after")
    def _check_years(self) -> Self:
        if self.end_year <= self.start_year:
            raise ValueError(
                f"end_year ({self.end_year}) must be after start_year ({self.start_year});"
                " end_year is exclusive"
            )
        return self


class CleanSettings(_Section):
    out_dir: RootPath = Field(default=Path("data/clean"), description="Cleaned CSVs and features.")


class ApiSettings(_Section):
    host: str = Field(default="127.0.0.1", description="Address `api` binds to.")
    port: int = Field(default=8000, ge=1, le=65535, description="Port `api` listens on.")


class Settings(BaseSettings):
    """Main settings. Priority: init args > env vars > .env > settings.toml > defaults."""

    model_config = SettingsConfigDict(
        toml_file=SETTINGS_FILE,
        env_file=".env",
        env_nested_delimiter="__",  # e.g. UNDERSTAT__START_YEAR=2018
        # Top-level keys owned by other parts of the project (such as [api_settings])
        # are left alone here; the sections above reject unknown keys.
        extra="ignore",
    )

    understat: UnderstatSettings = Field(default_factory=UnderstatSettings)
    clean: CleanSettings = Field(default_factory=CleanSettings)
    transfermarkt: TransfermarktSettings = Field(default_factory=TransfermarktSettings)
    api_settings: ApiSettings = Field(default_factory=ApiSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init args, env vars, .env, then settings.toml."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def _describe(err: ValidationError) -> str:
    lines = [f"Invalid settings ({SETTINGS_FILE or 'no settings.toml'}, env, .env):"]
    for e in err.errors():
        where = ".".join(str(part) for part in e["loc"])
        env_name = "__".join(str(part) for part in e["loc"]).upper()
        got = "" if e["type"] == "missing" else f" (got {e['input']!r})"
        lines.append(f"  {where}: {e['msg']}{got}  [env: {env_name}]")
    return "\n".join(lines)


def load_settings(**overrides: Any) -> Settings:
    """Settings from every source, with a readable ConfigError when a value is wrong."""
    try:
        return Settings(**overrides)
    except ValidationError as err:
        raise ConfigError(_describe(err)) from None


@cache
def get_settings() -> Settings:
    """The project settings, loaded once."""
    return load_settings()


def __getattr__(name: str) -> Settings:
    # `from money_printer_research.config import settings` loads on first use, so
    # importing this module never fails on a bad value; the import that needs it does.
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def show_config() -> None:
    """Console script: print the settings in effect as JSON."""
    print(f"# settings file: {SETTINGS_FILE or 'none (defaults and environment only)'}")
    print(f"# project root:  {ROOT}")
    print(json.dumps(get_settings().model_dump(mode="json"), indent=2, ensure_ascii=False))
