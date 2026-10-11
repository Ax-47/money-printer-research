"""Writer on top of Result: arrows that keep a log of what each step did.

Logged(value, log)   a value with the entries written so far; the log is a tuple,
                     a monoid under + with () as its unit
writer(f, note)      f as an arrow Logged[A] -> Result[Logged[B]]: run f on the value,
                     then append note(a, b) to the log
told(x)              Logged(x, ()): the start of a logged pipeline

writer returns a K, so logged steps compose with >> like any other arrow, and the
laws hold: (f >> g) >> h == f >> (g >> h) for both the value and the log, and
IDENTITY is still the unit. An Err stops the pipeline as usual; its reason names
the step.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .monoid import K
from .morphism import Morphism
from .result import Result, apply_lifted, name_of


@dataclass(frozen=True)
class Logged[T, W]:
    """A value and the log entries written while computing it, oldest first."""

    value: T
    log: tuple[W, ...] = ()

    def tell(self, *entries: W) -> "Logged[T, W]":
        """The same value with entries appended to the log."""
        return Logged(self.value, self.log + entries)


def told[T](value: T) -> Logged[T, Any]:
    """Start a logged pipeline: value with an empty log."""
    return Logged(value)


def writer[A, B, W](
    f: Morphism[A, Any], note: Callable[[A, B], W]
) -> K[Logged[A, W], Logged[B, W]]:
    """Lift f into the logged category: f on the value, then note(input, output) on the log.

    f may return a plain value or a Result (auto lift, as with K); an exception or
    an Err from f is the arrow's Err. note runs only on success.
    """

    def run(x: Logged[A, W]) -> Result[Logged[B, W]]:
        return apply_lifted(f, x.value).map(lambda b: Logged(b, x.log + (note(x.value, b),)))

    return K(run, name_of(f))


__all__ = ["Logged", "told", "writer"]
