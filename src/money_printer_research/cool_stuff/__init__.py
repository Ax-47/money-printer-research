"""Result, Kleisli arrows and Rust-style error handling, in one import.

    from money_printer_research.cool_stuff import Ok, Err, Result, monad, compose, lift

Modules, each building on the ones above it, all re-exported here:

    result    the data: Ok / Err with Rust's methods (map, and_then, unwrap, if_err, ...)
    morphism  arrow types: Morphism[A, B], ResultMorphism[A, B] (A -> Result[B])
    functor   lifting into Result: lift, catch, @fallible; functor over dicts
    monad     the Kleisli category: pure, bind, kleisli (>=>), @monad for .if_err(),
              all_ok (the applicative: independent steps, errors summed)
    monoid    K arrows under >> with IDENTITY as unit, compose (mconcat), fanout (&)
    validate  pandera schemas as steps: check(Model), @returns(Model)

Only validate imports pandera.
"""

from .functor import catch, fallible, functor, lift
from .monad import all_ok, bind, kleisli, monad, pure
from .monoid import IDENTITY, K, compose, fanout
from .morphism import Context, ContextMorphism, Morphism, ResultMorphism
from .result import EarlyReturn, Err, Ok, Result, ResultApi, UnwrapError, apply_lifted, name_of
from .validate import check, returns

__all__ = [
    "Context",
    "ContextMorphism",
    "EarlyReturn",
    "Err",
    "IDENTITY",
    "K",
    "Morphism",
    "Ok",
    "Result",
    "ResultApi",
    "ResultMorphism",
    "UnwrapError",
    "all_ok",
    "apply_lifted",
    "bind",
    "catch",
    "check",
    "compose",
    "fallible",
    "fanout",
    "functor",
    "kleisli",
    "lift",
    "monad",
    "name_of",
    "pure",
    "returns",
]
