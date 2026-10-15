"""Every method of Ok and Err, and the small combinators around them."""

import pandera.polars as pa
import polars as pl
import pytest
from pandera.typing import FieldType as F

from money_printer_research.cool_stuff import Err, Ok, catch, check
from money_printer_research.cool_stuff.functor import functor
from money_printer_research.cool_stuff.monoid import K, fanout
from money_printer_research.cool_stuff.result import UnwrapError


def test_ok_methods() -> None:
    ok = Ok(2)
    assert ok.is_ok() and not ok.is_err()
    assert ok.ok() == 2
    assert ok.map(lambda x: x + 1) == Ok(3)
    assert ok.map_err(str.upper) is ok
    assert ok.or_else(lambda e: Ok(0)) is ok
    assert ok.unwrap() == ok.expect("never") == ok.unwrap_or(9) == ok.unwrap_or_else(lambda e: 9)
    assert ok.if_err() == 2


def test_err_methods() -> None:
    cause = ValueError("boom")
    err = Err("bad", cause)
    assert err.is_err() and not err.is_ok()
    assert err.ok() is None
    assert err.map(lambda x: x) is err
    assert err.map_err(str.upper) == Err("BAD", cause)
    assert err.and_then(lambda x: Ok(x)) is err
    assert (err >> (lambda x: x)) is err
    assert err.or_else(lambda e: Ok(e.reason)) == Ok("bad")
    assert err.unwrap_or(1) == 1
    assert err.unwrap_or_else(lambda e: len(e.reason)) == 3
    with pytest.raises(UnwrapError, match="^bad$") as raised:
        err.unwrap()
    assert raised.value.__cause__ is cause
    with pytest.raises(UnwrapError, match="^step: bad$"):
        err.expect("step")


def test_functor_maps_dict_values() -> None:
    def twice(x: int) -> int:
        return 2 * x

    double = functor(twice)
    assert double({"a": 1, "b": 2}) == {"a": 2, "b": 4}
    assert double.__name__.startswith("functor(")


def test_catch_turns_an_exception_into_err() -> None:
    def fails() -> int:
        raise KeyError("x")

    out = catch(fails, "reading")
    assert out.is_err() and out.reason.startswith("reading: KeyError")  # pyright: ignore[reportAttributeAccessIssue]
    assert catch(lambda: Ok(1)) == Ok(1)


def test_k_fanout_map_and_repr() -> None:
    def plus_one(x: int) -> Ok[int]:
        return Ok(x + 1)

    def minus(x: int) -> Ok[int]:
        return Ok(-x)

    inc, neg = K(plus_one, "inc"), K(minus, "neg")
    assert (inc & neg).run(1) == Ok((2, -1))
    assert inc.map(str).run(1) == Ok("2")
    assert repr(inc) == "K(inc)"
    both_fail = fanout(lambda x: Err("left"), lambda x: Err("right")).run(0)
    assert both_fail.is_err() and [e.reason for e in both_fail.errors] == ["left", "right"]  # pyright: ignore[reportAttributeAccessIssue]


class _Positive(pa.DataFrameModel):
    x: F[int] = pa.Field(gt=0)


def test_check_returns_err_with_every_failure() -> None:
    out = check(_Positive).run(pl.DataFrame({"x": [1, -1, 0]}))
    assert out.is_err()
    assert out.reason.startswith("_Positive:")  # pyright: ignore[reportAttributeAccessIssue]
    assert check(_Positive).run(pl.DataFrame({"x": [1]})).is_ok()
