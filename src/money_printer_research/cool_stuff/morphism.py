"""Morphisms: the arrows of the categories in this package, as type aliases.

Morphism[A, B]          a plain function A -> B (an arrow of Python's types)
ResultMorphism[A, B]    A -> Result[B]: an arrow of the Kleisli category of Result
ContextMorphism[A, B]   the same, named for the effect: A -> Context[B]
"""

from collections.abc import Callable

from .result import Result

type Morphism[A, B] = Callable[[A], B]
type Context[B] = Result[B]  # the effect wrapping the output
type ContextMorphism[A, B] = Morphism[A, Context[B]]  # A -> Result[B]
type ResultMorphism[A, B] = Morphism[A, Result[B]]  # A -> Result[B]


__all__ = ["Context", "ContextMorphism", "Morphism", "ResultMorphism"]
