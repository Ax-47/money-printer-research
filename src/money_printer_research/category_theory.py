from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Err:
    reason: str
    cause: BaseException | None = None


type Result[T] = Ok[T] | Err
type Morphism[A, B] = Callable[[A], B]
type Context[B] = Result[B]  # the effect wrapping the output
type ContextMorphism[A, B] = Morphism[A, Context[B]]  # A -> Result[B]


def pure[T](x: T) -> Result[T]:
    """Identity morphism of the Kleisli category."""
    return Ok(x)


def bind[A, B](m: Result[A], f: Morphism[A, Result[B]]) -> Result[B]:
    match m:
        case Ok(value):
            return f(value)
        case Err():
            return m


def kleisli[A, B, C](
    f: Morphism[A, Result[B]], g: Morphism[B, Result[C]]
) -> Morphism[A, Result[C]]:
    """Kleisli composition: f >=> g."""
    return lambda a: bind(f(a), g)


def functor[A, B](f: Morphism[A, B]) -> Morphism[dict[str, A], dict[str, B]]:
    def mapped(xs: dict[str, A]) -> dict[str, B]:
        return {key: f(x) for key, x in xs.items()}

    mapped.__name__ = f"functor({getattr(f, '__name__', 'f')})"
    return mapped


def compose(*morphisms: Morphism[Any, Result[Any]]) -> Morphism[Any, Result[Any]]:
    return reduce(kleisli, morphisms, pure)


def lift[A, B](f: Morphism[A, B]) -> ContextMorphism[A, B]:
    """Turn a plain function into a Kleisli arrow; exceptions become Err."""
    name = getattr(f, "__name__", repr(f))

    def context_morphism(a: A) -> Result[B]:
        try:
            return Ok(f(a))
        except Exception as e:
            return Err(f"{name}: {e!r}", cause=e)

    return context_morphism


__all__ = [
    "Err",
    "Ok",
    "Result",
    "Morphism",
    "pure",
    "bind",
    "kleisli",
    "compose",
    "lift",
    "functor",
]
