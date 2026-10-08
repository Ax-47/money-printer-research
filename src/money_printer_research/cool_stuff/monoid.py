"""Kleisli arrows as a monoid: K under >>, with IDENTITY as the unit.

K(f, name)      an arrow A -> Result[B], callable like f
f >> g          Kleisli composition (f >=> g); associative
IDENTITY        the unit: IDENTITY >> f == f == f >> IDENTITY
compose(...)    mconcat: f1 >> f2 >> ..., typed for up to 8 steps
f & g / fanout  both on one input; Ok((b, c)) or one Err with every reason
f.map(h)        functor map over the output
"""

from functools import reduce
from typing import Any, overload

from .functor import lift
from .monad import all_ok, kleisli, pure
from .morphism import Morphism
from .result import Result, apply_lifted, name_of


class K[A, B]:
    """A Kleisli arrow A -> Result[B]: callable, and a monoid under >>.

    f >> g is Kleisli composition (f >=> g): run f, feed its Ok into g, stop at
    the first Err. It is associative, and IDENTITY (pure) is its unit:

        (f >> g) >> h == f >> (g >> h)
        IDENTITY >> f == f == f >> IDENTITY

    so arrows can be grouped and joined freely: compose(...) >> compose(...).

    Auto lift: K(f) and either side of >> take any function. One returning a
    Result is used as is; one returning anything else is lifted (value -> Ok,
    exception -> Err), so K(f) >> g >> check(M) needs no lift(...) anywhere.
    """

    __slots__ = ("name", "run")

    run: Morphism[A, Result[B]]
    name: str

    @overload
    def __init__(self, run: Morphism[A, Result[B]], name: str = "") -> None: ...
    @overload
    def __init__(self, run: Morphism[A, B], name: str = "") -> None: ...
    def __init__(self, run: Morphism[A, Any], name: str = "") -> None:
        """Wrap run as an arrow, lifting it if it returns a plain value.

        The name defaults to run's, so K(f) prints as f in pipeline names and errors.
        """
        self.run = _auto(run)
        self.name = name or name_of(run)

    def __call__(self, a: A) -> Result[B]:
        """Run the arrow on a."""
        return self.run(a)

    @overload
    def __rshift__[C](self, g: Morphism[B, Result[C]]) -> "K[A, C]": ...
    @overload
    def __rshift__[C](self, g: Morphism[B, C]) -> "K[A, C]": ...
    def __rshift__(self, g: Morphism[B, Any]) -> "K[A, Any]":
        """Compose: self, then g on its Ok value (>=>); a plain g is lifted."""
        return K(kleisli(self.run, _auto(g)), f"{self.name} >> {name_of(g)}")

    @overload
    def __rrshift__[Z](self, f: Morphism[Z, Result[A]]) -> "K[Z, B]": ...
    @overload
    def __rrshift__[Z](self, f: Morphism[Z, A]) -> "K[Z, B]": ...
    def __rrshift__(self, f: Morphism[Any, Any]) -> "K[Any, B]":
        """Compose with a function on the left: f >> self; a plain f is lifted."""
        return K(kleisli(_auto(f), self.run), f"{name_of(f)} >> {self.name}")

    def __and__[C](self, g: "K[A, C]") -> "K[A, tuple[B, C]]":
        """Fan out: self and g on the same input (&&&); see fanout."""
        return fanout(self, g)

    def map[C](self, f: Morphism[B, C]) -> "K[A, C]":
        """Apply plain f to the output (functor map); an exception in f becomes Err."""
        return K(kleisli(self.run, lift(f)), f"{self.name} >> {name_of(f)}")

    def __repr__(self) -> str:
        """Show the pipeline: K(f >> g >> check(M))."""
        return f"K({self.name})"

    @property
    def __name__(self) -> str:
        """The pipeline as text, for error messages and printing."""
        return self.name


def _auto(f: Morphism[Any, Any]) -> Morphism[Any, Result[Any]]:
    """F as a Kleisli step: a Result from f passes through, a plain value is lifted."""
    if isinstance(f, K):
        return f.run  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def step(x: Any) -> Result[Any]:
        return apply_lifted(f, x)

    step.__name__ = name_of(f)
    return step


def fanout[A, B, C](f: Morphism[A, Result[B]], g: Morphism[A, Result[C]]) -> K[A, tuple[B, C]]:
    """Run f and g on the same input (f &&& g); errors from both sides accumulate."""

    def run(a: A) -> Result[tuple[B, C]]:
        return all_ok(f(a), g(a))

    return K(run, f"({name_of(f)} & {name_of(g)})")


IDENTITY: K[Any, Any] = K(pure, "id")
"""The unit of >>: IDENTITY >> f == f == f >> IDENTITY."""


@overload
def compose() -> K[Any, Any]: ...


@overload
def compose[A, B](f1: Morphism[A, Result[B]], /) -> K[A, B]: ...


@overload
def compose[A, B, C](f1: Morphism[A, Result[B]], f2: Morphism[B, Result[C]], /) -> K[A, C]: ...


@overload
def compose[A, B, C, D](
    f1: Morphism[A, Result[B]], f2: Morphism[B, Result[C]], f3: Morphism[C, Result[D]], /
) -> K[A, D]: ...


@overload
def compose[A, B, C, D, E](
    f1: Morphism[A, Result[B]],
    f2: Morphism[B, Result[C]],
    f3: Morphism[C, Result[D]],
    f4: Morphism[D, Result[E]],
    /,
) -> K[A, E]: ...


@overload
def compose[A, B, C, D, E, F](
    f1: Morphism[A, Result[B]],
    f2: Morphism[B, Result[C]],
    f3: Morphism[C, Result[D]],
    f4: Morphism[D, Result[E]],
    f5: Morphism[E, Result[F]],
    /,
) -> K[A, F]: ...


@overload
def compose[A, B, C, D, E, F, G](
    f1: Morphism[A, Result[B]],
    f2: Morphism[B, Result[C]],
    f3: Morphism[C, Result[D]],
    f4: Morphism[D, Result[E]],
    f5: Morphism[E, Result[F]],
    f6: Morphism[F, Result[G]],
    /,
) -> K[A, G]: ...


@overload
def compose[A, B, C, D, E, F, G, H](
    f1: Morphism[A, Result[B]],
    f2: Morphism[B, Result[C]],
    f3: Morphism[C, Result[D]],
    f4: Morphism[D, Result[E]],
    f5: Morphism[E, Result[F]],
    f6: Morphism[F, Result[G]],
    f7: Morphism[G, Result[H]],
    /,
) -> K[A, H]: ...


@overload
def compose[A, B, C, D, E, F, G, H, I](
    f1: Morphism[A, Result[B]],
    f2: Morphism[B, Result[C]],
    f3: Morphism[C, Result[D]],
    f4: Morphism[D, Result[E]],
    f5: Morphism[E, Result[F]],
    f6: Morphism[F, Result[G]],
    f7: Morphism[G, Result[H]],
    f8: Morphism[H, Result[I]],
    /,
) -> K[A, I]: ...


def compose(*morphisms: Morphism[Any, Result[Any]]) -> K[Any, Any]:
    """Fold the arrows with >> from IDENTITY (mconcat): f1 >=> f2 >=> ...

    Typed for up to 8 steps: compose(lift(f), lift(g)) is K[A, C] when f: A -> B
    and g: B -> C, and a step whose input does not match the previous output is a
    type error. The result is a K, so compose(...) >> compose(...) works; for more
    than 8 steps, join groups that way or nest compose.
    """
    if not morphisms:
        return IDENTITY
    first, *rest = morphisms
    start = first if isinstance(first, K) else K(first, name_of(first))
    return reduce(K.__rshift__, rest, start)  # IDENTITY >> f == f, so skip it


__all__ = ["IDENTITY", "K", "compose", "fanout"]
