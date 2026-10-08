"""The Result monad: pure, bind, Kleisli composition, and Rust's `?` as @monad.

    pure(x)         Ok(x): the unit
    bind(m, f)      m >>= f: f on m's value; an Err passes through
    kleisli(f, g)   f >=> g: compose two A -> Result[B] arrows
    @monad          lets a function use r.if_err() (Rust's `?`)
    all_ok(ra, rb, ...)  Ok((a, b, ...)) if all are Ok, else every Err summed (e1 + e2)
                    (the applicative: independent steps, errors accumulate)

    @monad
    def clean(settings: Settings) -> Result[Cleaned]:
        xg = understat(settings).if_err()      # let xg = understat(settings)?;
        tm = transfermarkt(settings).if_err()  # let tm = transfermarkt(settings)?;
        return Ok(Cleaned(xg, tm))

`?` is spelled .if_err() since Python has no `?` operator. Only a function decorated
with @monad may call .if_err(); the decorator turns the early return into that Err.
"""

import operator
from collections.abc import Callable
from functools import reduce, wraps
from typing import Any, overload

from .morphism import Morphism, ResultMorphism
from .result import EarlyReturn, Err, Ok, Result


def pure[T](x: T) -> Result[T]:
    """Wrap x in Ok: the identity morphism of the Kleisli category."""
    return Ok(x)


def bind[A, B](m: Result[A], f: ResultMorphism[A, B]) -> Result[B]:
    """Feed m's value into f (>>=); an Err passes through unchanged."""
    return m.and_then(f)


def kleisli[A, B, C](
    f: Morphism[A, Result[B]], g: Morphism[B, Result[C]]
) -> Morphism[A, Result[C]]:
    """Kleisli composition: f >=> g."""
    return lambda a: bind(f(a), g)


def monad[**P, T](f: Callable[P, Result[T]]) -> Callable[P, Result[T]]:
    """Let f use .if_err() (Rust's `?`): an Err hit by .if_err() becomes f's return value."""

    @wraps(f)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result[T]:
        try:
            return f(*args, **kwargs)
        except EarlyReturn as early:
            return early.err

    return run


@overload
def all_ok[A](a: Result[A], /) -> Result[tuple[A]]: ...


@overload
def all_ok[A, B](a: Result[A], b: Result[B], /) -> Result[tuple[A, B]]: ...


@overload
def all_ok[A, B, C](a: Result[A], b: Result[B], c: Result[C], /) -> Result[tuple[A, B, C]]: ...


@overload
def all_ok[A, B, C, D](
    a: Result[A], b: Result[B], c: Result[C], d: Result[D], /
) -> Result[tuple[A, B, C, D]]: ...


@overload
def all_ok[A, B, C, D, E](
    a: Result[A], b: Result[B], c: Result[C], d: Result[D], e: Result[E], /
) -> Result[tuple[A, B, C, D, E]]: ...


@overload
def all_ok[A, B, C, D, E, F](
    a: Result[A], b: Result[B], c: Result[C], d: Result[D], e: Result[E], f: Result[F], /
) -> Result[tuple[A, B, C, D, E, F]]: ...


@overload
def all_ok(*results: Result[Any]) -> Result[tuple[Any, ...]]: ...


def all_ok(*results: Result[Any]) -> Result[tuple[Any, ...]]:
    """Ok of every value if all are Ok, else every Err summed into one (errors accumulate).

    The applicative of Result: the steps are independent, so all of them run and
    a failure in one does not hide a failure in another. Typed for up to 6.
    """
    errors = [r for r in results if isinstance(r, Err)]
    if errors:
        return reduce(operator.add, errors)
    return Ok(tuple(r.value for r in results if isinstance(r, Ok)))


__all__ = ["all_ok", "bind", "kleisli", "monad", "pure"]
