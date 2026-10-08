"""The laws behind category_theory, checked on random inputs with hypothesis.

functor: map(id) == id, map(g . f) == map(f) . map(g)
monad:   left identity, right identity, associativity of bind
monoid:  K under >> is associative with IDENTITY as unit; Err under + is associative
plus the behaviour the pipeline relies on: all_ok accumulates, .if_err() returns early
even through lift / catch / fallible / returns.
"""

from collections.abc import Callable

import pandera.polars as pa
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from money_printer_research.cool_stuff import (
    IDENTITY,
    Err,
    K,
    Ok,
    Result,
    all_ok,
    bind,
    catch,
    compose,
    fallible,
    lift,
    monad,
    pure,
    returns,
)

ints = st.integers(min_value=-1000, max_value=1000)
errs = st.text(min_size=1, max_size=20).map(Err)
results: st.SearchStrategy[Result[int]] = st.one_of(ints.map(Ok), errs)

PLAIN: list[Callable[[int], int]] = [lambda x: x + 1, lambda x: x * 2, lambda x: -x, abs]
FALLIBLE: list[Callable[[int], Result[int]]] = [
    lambda x: Ok(x + 1),
    lambda x: Err("negative") if x < 0 else Ok(x * 2),
    lambda x: Err("odd") if x % 2 else Ok(x // 2),
    pure,
]
plain = st.sampled_from(PLAIN)
fallible_fns = st.sampled_from(FALLIBLE)


def same[T](a: Result[T], b: Result[T]) -> bool:
    """Equal results; Errs compare by reason (causes are distinct exception objects)."""
    match a, b:
        case Ok(x), Ok(y):
            return x == y
        case Err(r1), Err(r2):
            return r1 == r2
        case _:
            return False


# ---- functor ------------------------------------------------------------------------


@given(results)
def test_functor_identity(r: Result[int]) -> None:
    """map(id) leaves a Result unchanged."""
    assert r.map(lambda x: x) == r


@given(results, plain, plain)
def test_functor_composition(
    r: Result[int], f: Callable[[int], int], g: Callable[[int], int]
) -> None:
    """Mapping g . f equals mapping f then g."""
    assert r.map(lambda x: g(f(x))) == r.map(f).map(g)


# ---- monad --------------------------------------------------------------------------


@given(ints, fallible_fns)
def test_monad_left_identity(a: int, f: Callable[[int], Result[int]]) -> None:
    """bind(pure(a), f) == f(a)."""
    assert bind(pure(a), f) == f(a)


@given(results)
def test_monad_right_identity(m: Result[int]) -> None:
    """bind(m, pure) == m."""
    assert bind(m, pure) == m


@given(results, fallible_fns, fallible_fns)
def test_monad_associativity(
    m: Result[int], f: Callable[[int], Result[int]], g: Callable[[int], Result[int]]
) -> None:
    """Binding f then g equals binding (f then g)."""
    assert bind(bind(m, f), g) == bind(m, lambda x: bind(f(x), g))


@given(results, fallible_fns, fallible_fns)
def test_rshift_is_bind(
    m: Result[int], f: Callable[[int], Result[int]], g: Callable[[int], Result[int]]
) -> None:
    """Operator >> on a Result is bind: m >> f >> g == bind(bind(m, f), g)."""
    assert m >> f >> g == bind(bind(m, f), g)


@given(results, plain)
def test_rshift_auto_lifts_plain_functions(m: Result[int], f: Callable[[int], int]) -> None:
    """A plain function after >> is lifted: m >> f == m.map(f)."""
    assert m >> f == m.map(f)


@given(ints)
def test_auto_lift_turns_exceptions_into_err(a: int) -> None:
    """An exception in an auto-lifted step becomes an Err naming the step."""

    def hundred_over(x: int) -> int:
        return 100 // x

    def plus_one(x: int) -> int:
        return x + 1

    out = Ok(a) >> hundred_over
    match out:
        case Ok(v):
            assert v == 100 // a
        case Err(reason):
            assert a == 0
            assert reason.startswith("hundred_over: ZeroDivisionError")
    assert same(K(hundred_over)(a), out)
    assert same((hundred_over >> K(plus_one))(a), out.map(plus_one))


# ---- monoid: K under >> ---------------------------------------------------------------


@given(ints, fallible_fns, fallible_fns, fallible_fns)
def test_kleisli_associativity(
    a: int,
    f: Callable[[int], Result[int]],
    g: Callable[[int], Result[int]],
    h: Callable[[int], Result[int]],
) -> None:
    """(f >> g) >> h equals f >> (g >> h)."""
    kf, kg, kh = K(f), K(g), K(h)
    assert same(((kf >> kg) >> kh)(a), (kf >> (kg >> kh))(a))


@given(ints, fallible_fns)
def test_kleisli_identity(a: int, f: Callable[[int], Result[int]]) -> None:
    """(f >> g) >> h equals f >> (g >> h)."""
    """Binding f then g equals binding (f then g)."""
    k = K(f)
    assert same((IDENTITY >> k)(a), k(a))
    assert same((k >> IDENTITY)(a), k(a))


@given(ints)
def test_compose_matches_rshift_with_exceptions(a: int) -> None:
    """bind(m, pure) == m."""

    def hundred_over(x: int) -> int:
        return 100 // x

    def plus_one(x: int) -> int:
        return x + 1

    div, inc = lift(hundred_over), lift(plus_one)
    assert same(compose(inc, div, inc)(a), (K(inc) >> div >> inc)(a))
    assert compose()(a) == Ok(a)


# ---- Err as a semigroup, all_ok as the applicative ------------------------------------


@given(errs, errs, errs)
def test_err_sum_associative(a: Err, b: Err, c: Err) -> None:
    """bind(pure(a), f) == f(a)."""
    assert (a + b) + c == a + (b + c)
    assert ((a + b) + c).errors == (a, b, c)


@given(st.lists(results, min_size=1, max_size=6))
def test_all_ok_accumulates(rs: list[Result[int]]) -> None:
    """Mapping g . f equals mapping f then g."""
    out = all_ok(*rs)
    errors = [r for r in rs if isinstance(r, Err)]
    if errors:
        assert isinstance(out, Err)
        assert out.errors == tuple(errors)
    else:
        assert out == Ok(tuple(r.unwrap() for r in rs))


# ---- .if_err() and @monad -----------------------------------------------------------


def test_if_err_returns_early() -> None:
    """map(id) leaves a Result unchanged."""
    ran: list[str] = []

    @monad
    def f() -> Result[int]:
        Err("stop").if_err()
        ran.append("after")
        return Ok(1)

    assert f() == Err("stop")
    assert ran == []


class _Frame(pa.DataFrameModel):
    x: int = pa.Field()


def test_if_err_passes_through_wrappers() -> None:
    """EarlyReturn is a BaseException: lift, catch, fallible and returns let it through."""
    inner = Err("inner")

    def raises_early(_: object = None) -> pl.DataFrame:
        inner.if_err()
        return pl.DataFrame({"x": [1]})

    wrappers = [
        lambda: lift(raises_early)(None),
        lambda: catch(raises_early),
        lambda: fallible(raises_early)(),
        lambda: returns(_Frame)(raises_early)(),
    ]
    for call in wrappers:

        @monad
        def outer() -> Result[int]:
            call()
            return Ok(1)

        assert outer() is inner
