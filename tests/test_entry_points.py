import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import money_printer_research
import money_printer_research.data_collection as data_collection
from money_printer_research import notebook
from money_printer_research.config import settings


def test_entry_point_loads_the_function_not_the_subpackage() -> None:
    # A fresh interpreter, as the console script runs: other tests here have already
    # imported the clean_data subpackage, which would bind it to the name first.
    code = (
        "import sys; from money_printer_research import clean_data; "
        "print(callable(clean_data), 'soccerdata' in sys.modules)"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    # The function, not the subpackage, and without loading the scraping libraries.
    assert run.stdout.split() == ["True", "False"]


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_thing"):
        _ = money_printer_research.no_such_thing  # pyright: ignore[reportAttributeAccessIssue]


def test_collect_data_runs_every_source_fastest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    for name in ("collect_schedule", "collect_player_stats", "collect_transfermarkt"):
        monkeypatch.setattr(data_collection, name, lambda s, name=name: order.append(name))

    data_collection.collect_data()

    assert order == ["collect_schedule", "collect_player_stats", "collect_transfermarkt"]


def test_collect_data_passes_the_global_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    for name in ("collect_schedule", "collect_player_stats", "collect_transfermarkt"):
        monkeypatch.setattr(data_collection, name, seen.append)
    data_collection.collect_data()
    assert seen == [settings] * 3


def test_notebook_root_is_the_project_root() -> None:
    assert (notebook.ROOT / "pyproject.toml").exists()
    assert (notebook.ROOT / "src" / "money_printer_research").is_dir()


def test_exec_notebook_opens_jupyterlab_at_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    argv: list[list[str]] = []

    def fake_main(args: list[str]) -> int:
        argv.append(args)
        return 0

    fake_labapp = ModuleType("jupyterlab.labapp")
    fake_labapp.main = fake_main  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "jupyterlab.labapp", fake_labapp)

    with pytest.raises(SystemExit) as exit_info:
        notebook.exec_notebook()

    assert exit_info.value.code == 0
    assert argv == [["--notebook-dir", str(Path(notebook.ROOT))]]


def test_clean_data_script_uses_the_wrapper_that_returns_none() -> None:
    # The generated script runs sys.exit(clean_data()): returning the cleaned frames
    # there printed them to stderr and exited with status 1.
    from importlib.metadata import entry_points

    scripts = {e.name: e.value for e in entry_points(group="console_scripts")}
    assert scripts["clean_data"] == "money_printer_research:exec_clean_data"


def test_exec_clean_data_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    import money_printer_research.clean_data as clean_data_module

    ran: list[bool] = []
    monkeypatch.setattr(clean_data_module, "clean_data", lambda: ran.append(True) or (1, 2))
    assert clean_data_module.exec_clean_data() is None
    assert ran == [True]
