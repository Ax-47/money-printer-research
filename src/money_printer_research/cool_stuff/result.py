"""Result in the style of Rust's std::result: Ok, Err and their methods.

    Ok(x) / Err(reason, cause)        pattern-match with `match`
    r.map(f)        Ok(f(x)), Err unchanged
    r.map_err(f)    Err(f(reason), cause), Ok unchanged
    r.and_then(f)   f(x) for f: T -> Result[U]  (bind, >>=)
    r >> f          bind as an operator, with auto lift: f may return a Result or a
                    plain value (wrapped in Ok); an exception in f becomes Err
    r.or_else(f)    f(err) on Err, Ok unchanged
    r.unwrap()      the value, or raise UnwrapError on Err (like a panic)
    r.expect(msg)   unwrap with your message
    r.unwrap_or(d) / r.unwrap_or_else(f)
    r.is_ok() / r.is_err() / r.ok()   ok() is the value or None
    r.if_err()      Rust's `?`: the value, or return this Err from the @monad function
    e1 + e2         one Err holding both (a semigroup); e.errors lists each one

ResultApi is the method set Ok and Err share, checked statically below like an
`impl Trait for` block, so a method added to one and not the other fails type checking.

The category theory built on it lives next door: morphism (arrow types),
functor (lift, catch, fallible), monad (pure, bind, kleisli, @monad, all_ok) and
monoid (K arrows under >>, IDENTITY, compose, fanout).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol, overload


@dataclass(frozen=True)
class Ok[T]:
    """The success case of Result: holds the value."""

    value: T

    def is_ok(self) -> bool:
        """Return True: this is an Ok."""
        return True

    def is_err(self) -> bool:
        """Return False: this is not an Err."""
        return False

    def ok(self) -> T | None:
        """Return the value (None only for an Err)."""
        return self.value

    def map[U](self, f: Callable[[T], U]) -> "Ok[U]":
        """Apply f to the value: Ok(f(x))."""
        return Ok(f(self.value))

    def map_err(self, f: Callable[[str], str]) -> "Ok[T]":
        """Return self unchanged; there is no error to map."""
        return self

    def and_then[U](self, f: "Callable[[T], Result[U]]") -> "Result[U]":
        """Chain a fallible step: f(x), itself a Result (bind, >>=)."""
        return f(self.value)

    @overload
    def __rshift__[U](self, f: "Callable[[T], Result[U]]") -> "Result[U]": ...
    @overload
    def __rshift__[U](self, f: Callable[[T], U]) -> "Result[U]": ...
    def __rshift__(self, f: Callable[[T], Any]) -> "Result[Any]":
        """Bind with auto lift: r >> f >> g chains; see apply_lifted."""
        return apply_lifted(f, self.value)

    def or_else(self, f: "Callable[[Err], Result[T]]") -> "Ok[T]":
        """Return self unchanged; the fallback runs only on Err."""
        return self

    def unwrap(self) -> T:
        """Return the value."""
        return self.value

    def expect(self, msg: str) -> T:
        """Return the value; msg is used only on Err."""
        return self.value

    def unwrap_or(self, default: object) -> T:
        """Return the value, ignoring the default."""
        return self.value

    def unwrap_or_else(self, f: "Callable[[Err], object]") -> T:
        """Return the value, without calling f."""
        return self.value

    def if_err(self) -> T:
        """Rust's `?` on Ok: the value."""
        return self.value


@dataclass(frozen=True)
class Err:
    """The failure case of Result: a readable reason and the exception behind it, if any."""

    reason: str
    cause: BaseException | None = None
    parts: tuple["Err", ...] = field(default=(), repr=False)
    """The single errors an Err made with + holds; empty for a single error."""

    @property
    def errors(self) -> tuple["Err", ...]:
        """Every single error in this Err: itself, or each part of a combined one."""
        return self.parts or (self,)

    def __add__(self, other: "Err") -> "Err":
        """Combine two errors into one (associative: a semigroup); reasons are joined."""
        return Err(
            f"{self.reason}\n\n{other.reason}",
            self.cause or other.cause,
            self.errors + other.errors,
        )

    def is_ok(self) -> bool:
        """Return False: this is not an Ok."""
        return False

    def is_err(self) -> bool:
        """Return True: this is an Err."""
        return True

    def ok(self) -> None:
        """Return None, as there is no value."""
        return None

    def map(self, f: Callable[[Any], Any]) -> "Err":
        """Return self unchanged; there is no value to map."""
        return self

    def map_err(self, f: Callable[[str], str]) -> "Err":
        """Apply f to the reason, keeping the cause."""
        return Err(f(self.reason), self.cause)

    def and_then(self, f: Callable[[Any], Any]) -> "Err":
        """Return self unchanged; the next step is skipped."""
        return self

    def __rshift__(self, f: Callable[[Any], Any]) -> "Err":
        """Bind on Err: the step is skipped, the Err passes on."""
        return self

    def or_else[T](self, f: "Callable[[Err], Result[T]]") -> "Result[T]":
        """Recover with f(self), itself a Result."""
        return f(self)

    def unwrap(self) -> NoReturn:
        """Raise UnwrapError with the reason (a panic); the cause is chained."""
        raise UnwrapError(self.reason) from self.cause

    def expect(self, msg: str) -> NoReturn:
        """Raise UnwrapError with msg and the reason; the cause is chained."""
        raise UnwrapError(f"{msg}: {self.reason}") from self.cause

    def unwrap_or[U](self, default: U) -> U:
        """Return the default."""
        return default

    def unwrap_or_else[U](self, f: "Callable[[Err], U]") -> U:
        """Return f(self), computed from the error."""
        return f(self)

    def if_err(self) -> NoReturn:
        """Rust's `?` on Err: return this Err from the enclosing @monad function."""
        raise EarlyReturn(self)


type Result[T] = Ok[T] | Err


class UnwrapError(Exception):
    """unwrap() or expect() on an Err: the Python form of a panic."""


class EarlyReturn(BaseException):  # noqa: N818 - control flow, not an error
    """Raised by Err.if_err(), caught by @monad: an early `return err`, never an error.

    A BaseException, like KeyboardInterrupt, so `except Exception` in lift, catch,
    @fallible or @returns lets it through to the @monad that called .if_err().
    """

    err: Err

    def __init__(self, err: Err) -> None:
        """Carry err to the enclosing @monad."""
        super().__init__(err.reason)
        self.err = err


class ResultApi[T](Protocol):
    """The methods every Result has, on Ok and Err alike (Rust's Result API)."""

    def is_ok(self) -> bool:
        """Return True for Ok."""
        ...

    def is_err(self) -> bool:
        """Return True for Err."""
        ...

    def ok(self) -> T | None:
        """Return the value, or None."""
        ...

    def map[U](self, f: Callable[[T], U]) -> Result[U]:
        """Apply f to an Ok's value."""
        ...

    def map_err(self, f: Callable[[str], str]) -> Result[T]:
        """Apply f to an Err's reason."""
        ...

    def and_then[U](self, f: Callable[[T], Result[U]]) -> Result[U]:
        """Chain a fallible step (bind)."""
        ...

    def __rshift__(self, f: Callable[[T], Any]) -> Result[Any]:
        """Bind with auto lift: r >> f."""
        ...

    def or_else(self, f: Callable[[Err], Result[T]]) -> Result[T]:
        """Recover from an Err."""
        ...

    def unwrap(self) -> T:
        """Return the value, or raise UnwrapError."""
        ...

    def expect(self, msg: str) -> T:
        """Unwrap, with a message on Err."""
        ...

    def unwrap_or(self, default: T) -> T:
        """Return the value, or the default."""
        ...

    def unwrap_or_else(self, f: Callable[[Err], T]) -> T:
        """Return the value, or f(err)."""
        ...

    def if_err(self) -> T:
        """Rust's `?` (inside @monad)."""
        ...


# Static checks, like `impl ResultApi for Ok` and `for Err`: fail if one lacks a method.
_OK_IMPLEMENTS_API: ResultApi[int] = Ok(0)
_ERR_IMPLEMENTS_API: ResultApi[int] = Err("")


def apply_lifted(f: Callable[[Any], Any], x: Any) -> Result[Any]:
    """Call f(x) as a Kleisli step, lifting it if needed (the auto lift behind >>).

    A Result from f is returned as is (bind); any other value v becomes Ok(v)
    (lift); an exception becomes Err("<name of f>: ...").
    """
    try:
        out = f(x)
    except Exception as e:
        return Err(f"{name_of(f)}: {e!r}", cause=e)
    return out if isinstance(out, Ok | Err) else Ok(out)


def name_of(f: object) -> str:
    """Return f's name for messages, or repr(f) for a callable without __name__."""
    return getattr(f, "__name__", None) or repr(f)


__all__ = [
    "EarlyReturn",
    "Err",
    "Ok",
    "Result",
    "ResultApi",
    "UnwrapError",
    "apply_lifted",
    "name_of",
]
