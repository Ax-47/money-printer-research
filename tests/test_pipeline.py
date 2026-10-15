import pytest

from money_printer_research import pipeline


def _record(monkeypatch: pytest.MonkeyPatch, fail: str | None = None) -> list[str]:
    ran: list[str] = []

    def fake_step(name: str):
        def step() -> None:
            ran.append(name)
            if name == "check_clean":
                raise SystemExit(0)  # what check_clean does when every file passes
            if name == fail:
                raise RuntimeError(name)

        return step

    monkeypatch.setattr(pipeline, "_step", fake_step)
    return ran


def test_runs_every_step_in_order_past_a_passing_check_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _record(monkeypatch)
    pipeline.run()
    assert ran == pipeline.STEPS


def test_from_starts_at_the_named_step(monkeypatch: pytest.MonkeyPatch) -> None:
    ran = _record(monkeypatch)
    pipeline.run("predict")
    assert ran == ["predict", "backtest", "report"]


def test_stops_at_the_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    ran = _record(monkeypatch, fail="modeling")
    with pytest.raises(RuntimeError):
        pipeline.run()
    assert ran[-1] == "modeling"


def test_check_clean_failing_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline, "_step", lambda name: lambda: (_ for _ in ()).throw(SystemExit(1))
    )
    with pytest.raises(SystemExit):
        pipeline.run("check_clean")


def test_every_step_resolves_to_a_callable() -> None:
    for name in pipeline.STEPS:
        assert callable(pipeline._step(name))  # pyright: ignore[reportPrivateUsage]


def test_console_script_is_registered() -> None:
    from importlib.metadata import entry_points

    scripts = {e.name: e.value for e in entry_points(group="console_scripts")}
    assert scripts["pipeline"] == "money_printer_research:exec_pipeline"
