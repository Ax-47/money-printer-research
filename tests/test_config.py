import json
from pathlib import Path

import pytest

from money_printer_research import config
from money_printer_research.config import (
    ROOT,
    ConfigError,
    Settings,
    UnderstatSettings,
    find_settings_file,
    load_settings,
)


def test_find_settings_file_from_env(tmp_path: Path) -> None:
    file = tmp_path / "custom.toml"
    file.write_text("")
    assert find_settings_file({"MPR_SETTINGS_FILE": str(file)}) == file


def test_find_settings_file_from_env_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="MPR_SETTINGS_FILE"):
        find_settings_file({"MPR_SETTINGS_FILE": str(tmp_path / "nope.toml")})


def test_find_settings_file_walks_up_from_cwd(tmp_path: Path) -> None:
    (tmp_path / "settings.toml").write_text("")
    nested = tmp_path / "notebooks" / "deep"
    nested.mkdir(parents=True)
    assert find_settings_file({}, cwd=nested) == tmp_path / "settings.toml"


def test_relative_paths_resolve_against_root() -> None:
    cfg = UnderstatSettings(out_dir=Path("somewhere"))
    assert cfg.out_dir == ROOT / "somewhere"


def test_env_overrides_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNDERSTAT__START_YEAR", "2018")
    monkeypatch.setenv("CLEAN__OUT_DIR", "/tmp/clean")
    s = Settings()
    assert s.understat.start_year == 2018
    assert s.clean.out_dir == Path("/tmp/clean")
    # Untouched values still come from settings.toml / defaults.
    assert s.understat.end_year > 2018


def test_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("UNDERSTAT__START_YEAR=2016\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().understat.start_year == 2016


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ConfigError, match=r"understat\.start_yeer: Extra inputs"):
        load_settings(understat={"start_yeer": 2018})


def test_error_names_the_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNDERSTAT__START_YEAR", "soon")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    message = str(exc.value)
    assert "understat.start_year" in message
    assert "'soon'" in message
    assert "UNDERSTAT__START_YEAR" in message


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"start_year": 2020, "end_year": 2020}, "end_year .* must be after start_year"),
        ({"start_year": 2010}, "greater than or equal to 2014"),
        ({"leagues": []}, "at least 1 item"),
        ({"leagues": ["ESP-La Liga", "ESP-La Liga"]}, "listed more than once: ESP-La Liga"),
    ],
)
def test_understat_rules(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        load_settings(understat=overrides)


def test_transfermarkt_ids_must_be_plain() -> None:
    with pytest.raises(ConfigError, match="not a Transfermarkt competition id"):
        load_settings(transfermarkt={"leagues": {"GB1'; DROP": "ENG-Premier League"}})


def test_settings_attribute_is_loaded_once() -> None:
    assert config.settings is config.settings
    assert isinstance(config.settings, Settings)


def test_unknown_module_attribute() -> None:
    with pytest.raises(AttributeError):
        _ = config.nope  # pyright: ignore[reportAttributeAccessIssue]


def test_show_config(capsys: pytest.CaptureFixture[str]) -> None:
    config.show_config()
    out = capsys.readouterr().out
    body = "\n".join(line for line in out.splitlines() if not line.startswith("#"))
    assert set(json.loads(body)) >= {"understat", "clean", "transfermarkt"}
