"""`uv run pipeline`: every step from the raw downloads to the report, in order.

    collect_data -> clean_data -> check_clean -> modeling -> predict -> backtest -> report

modeling also writes the learning curve, and report reruns market_test, strategies,
staking and practicality, so those are not separate steps. feature_importance (the
selection is fixed in SELECTED) and elo_sensitivity (about 18 minutes, its own seasons)
are left to their own commands.

`uv run pipeline --from modeling` starts at a step, for when the data is already there.
The run stops at the first step that fails.
"""

import argparse
import time
from collections.abc import Callable

STEPS = ["collect_data", "clean_data", "check_clean", "modeling", "predict", "backtest", "report"]


def _step(name: str) -> Callable[[], object]:
    """The function behind a step's console script, imported only when it runs."""
    import money_printer_research
    from money_printer_research.schema import check_clean

    if name == "check_clean":
        return check_clean
    attr = name if name == "collect_data" else f"exec_{name}"
    return getattr(money_printer_research, attr)


def _run(name: str) -> None:
    """Run one step; check_clean's exit status 0 means it passed, not that the run ends."""
    try:
        _step(name)()
    except SystemExit as e:
        if e.code not in (0, None):
            raise


def run(start: str = STEPS[0]) -> None:
    """Every step from `start` on, with a header and the time each one took."""
    steps = STEPS[STEPS.index(start) :]
    began = time.monotonic()
    for i, name in enumerate(steps, 1):
        print(f"\n== [{i}/{len(steps)}] {name} ==", flush=True)
        t = time.monotonic()
        _run(name)
        print(f"== {name} done in {time.monotonic() - t:.0f}s ==", flush=True)
    print(f"\npipeline done in {time.monotonic() - began:.0f}s", flush=True)


def exec_pipeline() -> None:
    """Entry point `pipeline`."""
    parser = argparse.ArgumentParser(
        prog="pipeline", description="Every step from the raw downloads to the report, in order."
    )
    parser.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    run(parser.parse_args().start)
