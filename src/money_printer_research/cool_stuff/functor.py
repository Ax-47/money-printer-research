"""Functors: lifting plain functions into a context while keeping their shape.

    functor(f)      dict[str, A] -> dict[str, B]: f over every value (the dict functor)
    lift(f)         A -> B  becomes  A -> Result[B]; an exception becomes Err
    catch(lambda: ...)   run code that may raise; Ok(value) or Err(exception)
    @fallible       lift for functions of any arguments

(The Result functor's own map is the method r.map(f) in result; K.map in monoid.)
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, overload

from .morphism import ContextMorphism, Morphism
from .result import Err, Ok, Result, apply_lifted, name_of


def functor[A, B](f: Morphism[A, B]) -> Morphism[dict[str, A], dict[str, B]]:
    """Map f over every value of a dict, keeping the keys."""

    def mapped(xs: dict[str, A]) -> dict[str, B]:
        return {key: f(x) for key, x in xs.items()}

    mapped.__name__ = f"functor({name_of(f)})"
    return mapped


@overload
def lift[A, B](f: Morphism[A, Result[B]]) -> ContextMorphism[A, B]: ...
@overload
def lift[A, B](f: Morphism[A, B]) -> ContextMorphism[A, B]: ...
def lift(f: Morphism[Any, Any]) -> ContextMorphism[Any, Any]:
    """Turn a function into a Kleisli arrow; exceptions become Err.

    A plain result is wrapped in Ok; a Result is passed on as is (auto lift).
    """

    def context_morphism(a: Any) -> Result[Any]:
        return apply_lifted(f, a)

    context_morphism.__name__ = name_of(f)
    return context_morphism


@overload
def catch[T](f: Callable[[], Result[T]], name: str | None = None) -> Result[T]: ...
@overload
def catch[T](f: Callable[[], T], name: str | None = None) -> Result[T]: ...
def catch(f: Callable[[], Any], name: str | None = None) -> Result[Any]:
    """Run f(): a plain value as Ok, a Result as is, an exception as Err("<name>: ...")."""
    label = name or name_of(f)
    try:
        out = f()
    except Exception as e:
        return Err(f"{label}: {e!r}", cause=e)
    return out if isinstance(out, Ok | Err) else Ok(out)


@overload
def fallible[**P, T](f: Callable[P, Result[T]]) -> Callable[P, Result[T]]: ...
@overload
def fallible[**P, T](f: Callable[P, T]) -> Callable[P, Result[T]]: ...
def fallible[**P](f: Callable[P, Any]) -> Callable[P, Result[Any]]:
    """Make f return Result like a Rust fn: a value as Ok, an exception as Err.

    If f already returns a Result it is passed on as is, so fallible is safe to
    put on any function.
    """
    label = name_of(f)

    @wraps(f)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result[Any]:
        return catch(lambda: f(*args, **kwargs), label)

    return run


__all__ = ["catch", "fallible", "functor", "lift"]
