"""One NEP 50 weak-scalar rule in every lifetime of a program.

A value is weak exactly when eager Python holds a Python scalar there: Python
scalar inputs, Python operators applied only to weak operands, the real and
imag parts that NumPy reads from them, and a diff of order zero, which NumPy
returns unchanged. Other NumPy and Array API functions return strong values.
Selecting a Python scalar for differentiation, staging, or calling a staged
program inside another trace must not change the dtype or value that eager
NumPy computes.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, example, given, reject, settings
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


@ad.primitive(name="tests.weak_scalar_lifetimes.double")
def _double(value: Any) -> Any:
    return value + value


@_double.def_abstract
def _double_abstract(value: Any) -> Any:
    spec = value.spec
    # Python adds two bools as integers; NumPy adds boolean arrays as a logical or.
    if spec.weak and str(spec.dtype) == "bool":
        return ad.ArraySpec((), "int64", weak=True)
    return spec


@_double.def_jvp
def _double_jvp(_output: Any, _primals: Any, tangents: Any) -> Any:
    (tangent,) = tangents
    return tangent + tangent


# The residual's state Jacobian is the identity, so a solve returns its right side.
_identity_root = ad.implicit_root(
    lambda solution, params: solution - params,
    solve=lambda residual, initial: initial - residual(initial),
    linear_solve=lambda _operator, rhs: rhs,
)

_UNARY: dict[str, Callable[[Any], Any]] = {
    "neg": operator.neg,
    "abs": abs,
    "square": lambda value: value**2,
    "floordiv": lambda value: value // 0.75,
    "mod": lambda value: value % 0.75,
    "real": lambda value: value.real,
    "sin": np.sin,
    "tanh": np.tanh,
    "sum": np.sum,
    "max": np.max,
    "prod": np.prod,
    "np.copy": np.copy,
    "np.real": np.real,
    # Advect's API functions follow the rule too: array and asarray are strong,
    # as NumPy's are, and the others return what eager evaluation of their
    # function does.
    "ad.array": ad.array,
    "ad.asarray": ad.asarray,
    "ad.checkpoint": ad.checkpoint(lambda value: value * 0.5),
    "ad.implicit_root": lambda value: _identity_root(value, initial=value),
    "ad.primitive": _double,
    "ad.stop_gradient": ad.stop_gradient,
}
# API functions that support only concrete dynamic traces, which staging rejects.
_DYNAMIC_ONLY = frozenset({"ad.checkpoint", "ad.implicit_root", "ad.stop_gradient"})
_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": lambda left, right: left / (abs(right) + 0.5),
    "lt": operator.lt,
    "np.add": np.add,
    "np.multiply": np.multiply,
}
# Array API functions of the namespace that the float32 input reports: NumPy
# eagerly, and Advect's staged namespace inside a stage.
_NAMESPACE_UNARY = ("xp.abs", "xp.conj", "xp.negative", "xp.real", "xp.sin")
_NAMESPACE_BINARY = ("xp.add", "xp.multiply")
_CONSTANTS = (0.5, 2, np.float32(1.5), np.float64(0.25))
_INPUTS = ("s", "x32", "x64")

_Expression = tuple[Any, ...]


def _reads_an_input(expression: _Expression) -> bool:
    name, *operands = expression
    return name in _INPUTS or (
        name != "const" and any(_reads_an_input(operand) for operand in operands)
    )


def _calls(expression: _Expression, functions: frozenset[str]) -> bool:
    """Return whether the expression calls one of *functions*."""
    name, *operands = expression
    return name != "const" and (
        name in functions or any(_calls(operand, functions) for operand in operands)
    )


def _expressions() -> st.SearchStrategy[_Expression]:
    leaves = st.one_of(
        st.sampled_from([(name,) for name in _INPUTS]),
        st.sampled_from(_CONSTANTS).map(lambda value: ("const", value)),
    )
    # A transform returns an input-independent constant through its own
    # output boundary, which this property does not exercise.
    return st.recursive(
        leaves,
        lambda children: st.one_of(
            st.tuples(st.sampled_from(sorted(_UNARY) + list(_NAMESPACE_UNARY)), children),
            st.tuples(
                st.sampled_from(sorted(_BINARY) + list(_NAMESPACE_BINARY)), children, children
            ),
        ),
        max_leaves=6,
    ).filter(_reads_an_input)


def _evaluate(expression: _Expression, inputs: dict[str, Any]) -> Any:
    name, *operands = expression
    if name in inputs:
        return inputs[name]
    if name == "const":
        return operands[0]
    values = [_evaluate(operand, inputs) for operand in operands]
    if name.startswith("xp."):
        namespace = inputs["x32"].__array_namespace__()
        return getattr(namespace, name.removeprefix("xp."))(*values)
    return (_UNARY if len(values) == 1 else _BINARY)[name](*values)


def _program(expression: _Expression) -> Callable[..., Any]:
    def program(s: Any, x32: Any, x64: Any) -> Any:
        return _evaluate(expression, {"s": s, "x32": x32, "x64": x64})

    return program


def _category(value: object) -> tuple[str, tuple[int, ...]]:
    """Return a Python scalar's type name, else a strong value's dtype, with its shape."""
    if type(value) in {bool, int, float, complex}:
        return type(value).__name__, ()
    array = np.asarray(value)
    return str(array.dtype), array.shape


def _lifetimes(
    program: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    staged: bool = True,
) -> dict[str, Any]:
    tangents = (1.0, np.ones(3, np.float32), np.ones(3))

    def traced(function: Callable[..., Any], *values: Any) -> Any:
        return ad.jvp(function, argnums=(0, 1, 2))(*values, tangents=tangents)[0]

    results = {
        "dynamic, scalar selected": traced(program, *args),
        "dynamic, scalar held": ad.jvp(program, argnums=(1, 2))(*args, tangents=tangents[1:])[0],
    }
    if not staged:
        return results
    program_stage = ad.stage(program, *args)
    return results | {
        "staged": program_stage(*args),
        "staged program in a dynamic trace": traced(program_stage, *args),
        "staged program in a stage": ad.stage(
            lambda *values: program_stage(*values),  # noqa: PLW0108 - explicit trace boundary
            *args,
        )(*args),
        "dynamic trace in a stage": ad.stage(lambda *values: traced(program, *values), *args)(
            *args
        ),
    }


@given(
    expression=_expressions(),
    s=st.floats(min_value=0.25, max_value=1.5),
)
# Issue examples: a NumPy or Array API function of the selected scalar is
# strong, and a Python operator or NumPy's real part keeps it weak.
@example(expression=("mul", ("sin", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("np.multiply", ("s",), ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("xp.multiply", ("s",), ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("xp.conj", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("mul", ("s",), ("s",)), ("x32",)), s=0.5)
@example(expression=("add", ("neg", ("s",)), ("const", np.float32(1.5))), s=0.5)
@example(expression=("mul", ("add", ("np.real", ("s",)), ("const", 0.5)), ("x32",)), s=0.5)
# ad.asarray of a selected scalar stayed weak; stop_gradient made it strong;
# a traced call of a checkpoint, a primitive or an implicit root on a held
# scalar raised "A traced primitive returned a scalar without an array provider".
@example(expression=("mul", ("ad.asarray", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("ad.stop_gradient", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("ad.checkpoint", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("ad.primitive", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("ad.implicit_root", ("s",)), ("x32",)), s=0.5)
# The test primitive declared a Python bool doubled as a bool, but Python adds
# two bools as an int: "declared shape=(), dtype=bool; produced Python int".
@example(expression=("add", ("s",), ("ad.primitive", ("lt", ("s",), ("s",)))), s=1.0)
# Inside a stage, a dynamic trace's Array API function lifted its NumPy tracer
# as a constant ("Could not encode staged constant element from TracedArray"),
# and its NumPy function read a staged constant as an ndarray, wrapped it as a
# tracer, or let the constant lift the tracer.
@example(expression=("xp.add", ("x32",), ("sin", ("x32",))), s=0.5)
@example(expression=("mul", ("sin", ("x32",)), ("xp.abs", ("const", 0.5))), s=0.5)
@example(expression=("np.add", ("s",), ("xp.abs", ("const", 0.5))), s=0.5)
@example(expression=("np.add", ("xp.abs", ("const", 0.5)), ("s",)), s=0.5)
# There, a primitive call on constants returns a NumPy tracer of a constant
# node, which an Array API function tried to encode as a staged constant.
@example(expression=("xp.add", ("s",), ("ad.primitive", ("const", np.float32(1.5)))), s=0.5)
# NumPy squares a boolean array to int8, but inside a stage a dynamic trace
# raised it to the power 2, which gives int64.
@example(expression=("square", ("lt", ("x32",), ("s",))), s=0.5)
# Staged replay copied a Python scalar input with its missing copy method.
@example(expression=("mul", ("np.copy", ("s",)), ("x32",)), s=0.5)
# The max and prod tangents of a selected scalar read the axes of its Python
# float tangent: "'float' object has no attribute 'ndim'".
@example(expression=("mul", ("max", ("s",)), ("x32",)), s=0.5)
@example(expression=("mul", ("prod", ("s",)), ("x32",)), s=0.5)
@settings(deadline=None)
def test_every_lifetime_promotes_python_scalars_as_eager_numpy(
    expression: _Expression,
    s: float,
) -> None:
    program = _program(expression)
    args = (s, np.array([0.5, -1.0, 2.0], np.float32), np.array([0.25, 1.5, -0.75]))

    try:
        expected = program(*args)
    except (ArithmeticError, TypeError, RuntimeWarning):
        # NumPy rejects the program, such as the negation of a boolean array.
        reject()
    # A transform returns a Python bool or int through its own float output
    # boundary, which this property does not exercise.
    assume(type(expected) not in {bool, int})
    category = _category(expected)
    tolerance = 4 * float(np.finfo(np.float32 if category[0] == "float32" else np.float64).eps)
    lifetimes = _lifetimes(program, args, staged=not _calls(expression, _DYNAMIC_ONLY))
    for lifetime, result in lifetimes.items():
        assert _category(result) == category, lifetime
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=tolerance, err_msg=lifetime)


# NumPy functions of any rank. A NumPy function reads a Python scalar as a
# rank-zero float64 array, so a selected scalar differentiates as that array.
_SHAPE_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "reshape": lambda value: np.reshape(value, (-1, 1)),
    "expand_dims": lambda value: np.expand_dims(value, 0),
    "squeeze": np.squeeze,
    "broadcast_to": lambda value: np.broadcast_to(value, (2, *getattr(value, "shape", ()))),
    "transpose": np.transpose,
    "stack": lambda value: np.stack([value, value]),
    "concatenate": lambda value: np.concatenate([value, value], axis=None),
    "tile": lambda value: np.tile(value, 2),
    "take": lambda value: np.take(value, [0, 0]),
    "cumsum": np.cumsum,
    "sort": lambda value: np.sort(value, axis=None),
    "max": np.max,
    "nanmean": np.nanmean,
    "where": lambda value: np.where(value > 0.5, value, -value),
    # NumPy's default norm flattens a value of any rank.
    "norm": np.linalg.norm,
    "sin": np.sin,
    "square": lambda value: value * value,
}
# These lack an abstract staging rule, or their staged form rejects rank zero.
_DYNAMIC_SHAPE_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "ravel": np.ravel,
    "atleast_1d": np.atleast_1d,
    "atleast_2d": np.atleast_2d,
    "outer": lambda value: np.outer(value, value),
    "inner": lambda value: np.inner(value, value),
    "kron": lambda value: np.kron(value, value),
    "pad": lambda value: np.pad(value, 1),
    "median": np.median,
    "einsum": lambda value: np.einsum("...,...->...", value, value),
    "diff": lambda value: np.diff(value, n=0),
}
_SHAPE_FUNCTIONS |= _DYNAMIC_SHAPE_FUNCTIONS


@given(
    names=st.lists(st.sampled_from(sorted(_SHAPE_FUNCTIONS)), min_size=1, max_size=3),
    s=st.floats(min_value=0.25, max_value=1.5),
)
# A rule read the shape, rank or dtype of the selected scalar's Python float
# primal: "'float' object has no attribute 'shape'".
@example(names=["reshape"], s=0.5)
@example(names=["expand_dims"], s=0.5)
@example(names=["squeeze"], s=0.5)
@example(names=["broadcast_to"], s=0.5)
@example(names=["take"], s=0.5)
@example(names=["nanmean"], s=0.5)
@example(names=["ravel"], s=0.5)
@example(names=["atleast_1d"], s=0.5)
@example(names=["atleast_1d", "sort"], s=0.5)
@example(names=["atleast_1d", "concatenate"], s=0.5)
@example(names=["atleast_2d"], s=0.5)
@example(names=["inner"], s=0.5)
@example(names=["kron"], s=0.5)
@example(names=["pad"], s=0.5)
@example(names=["median"], s=0.5)
# The first derivative held, and the second read the primal's shape.
@example(names=["outer"], s=0.5)
# The einsum handler read the shape of the scalar's Python value.
@example(names=["einsum"], s=0.5)
# A diff of order zero recorded the Python value as a strong result, whose
# shape the second derivative read.
@example(names=["diff"], s=0.5)
# Staging required axis= for the default norm of a rank-zero value, which
# NumPy flattens.
@example(names=["norm"], s=0.5)
@settings(deadline=None)
def test_a_selected_python_scalar_differentiates_through_shapes_as_numpy_reads_it(
    names: list[str],
    s: float,
) -> None:
    def program(value: Any) -> Any:
        for name in names:
            value = _SHAPE_FUNCTIONS[name](value)
        return np.sum(value)

    array = np.asarray(s)
    gradient = ad.grad(program)(s)
    _value, tangent = ad.jvp(program)(s, tangents=1.0)
    curvature = ad.hessian(program)(s)

    assert type(gradient) is float
    assert_allclose(gradient, ad.grad(program)(array), rtol=1e-12)
    assert_allclose(tangent, ad.jvp(program)(array, tangents=np.asarray(1.0))[1], rtol=1e-12)
    assert_allclose(curvature, ad.hessian(program)(array), rtol=1e-12)
    if _DYNAMIC_SHAPE_FUNCTIONS.keys().isdisjoint(names):
        staged = ad.stage(program, s)
        staged_gradient = ad.grad(staged)(s)
        assert type(staged_gradient) is float
        assert_allclose(staged(s), program(array), rtol=1e-12)
        assert_allclose(staged_gradient, gradient, rtol=1e-12)
        assert_allclose(ad.stage(ad.grad(program), s)(s), gradient, rtol=1e-12)


def test_a_diff_of_order_zero_keeps_a_python_scalar_weak_as_numpy_does() -> None:
    # NumPy returns the input of a diff of order zero itself. The trace recorded
    # a strong result that held the Python scalar, whose shape the second
    # derivative read: "NumPy attempted to convert a live Advect value into an
    # ndarray".
    x32 = np.array([0.5, -1.0, 2.0], np.float32)

    def program(s: Any) -> Any:
        return np.diff(s, n=0) * x32

    value, tangent = ad.jvp(program)(0.7, tangents=1.0)
    identity, _tangent = ad.jvp(lambda s: np.diff(s, n=0))(0.7, tangents=1.0)

    assert value.dtype == tangent.dtype == program(0.7).dtype == np.float32
    assert type(identity) is float
    assert_allclose(
        ad.hessian(lambda s: np.sum(program(s) ** 2))(0.7), 2 * np.sum(x32**2), rtol=1e-6
    )


def test_a_stage_of_python_scalars_alone_targets_the_numpy_revision() -> None:
    # Python scalars alone run on NumPy, but staging them compiled Advect's
    # latest revision, which NumPy 2.0-2.2 cannot run: "Staged profile
    # 'advect-array-1' requires Array API 2024.12; the runtime provider exposes
    # '2023.12'".
    program = ad.stage(lambda s: np.sum(np.reshape(s, (1,))) * 2.0, 0.5)

    assert program.array_api_version == ad.stage(np.sin, np.asarray(0.5)).array_api_version
    assert program(0.5) == 1.0
    assert ad.grad(program)(0.5) == 2.0


def test_derivatives_through_a_staged_python_scalar_product_match_eager() -> None:
    # np.multiply of a Python scalar is strong. Replayed in a dynamic trace that
    # selects the scalar, it became weak: "declared dtype=float64; produced float32".
    x = np.array([0.5, -1.0, 2.0], np.float32)
    program = ad.stage(lambda s, x: np.multiply(s, s) * x, 0.5, x)

    gradient = ad.grad(lambda s, x: np.sum(program(s, x)), argnums=0)(0.5, x)
    value, tangent = ad.jvp(program, argnums=0)(0.5, x, tangents=1.0)

    assert type(gradient) is float
    assert gradient == pytest.approx(2 * 0.5 * float(np.sum(x.astype(np.float64))))
    assert value.dtype == tangent.dtype == np.float64
    assert_allclose(tangent, x.astype(np.float64))


def test_a_nested_stage_calls_a_staged_numpy_function_of_a_python_scalar() -> None:
    # The staged program marked np.sin(s) weak by its input lineage, so the
    # enclosing stage "declared float32; produced float64".
    x = np.array([0.5, -1.0, 2.0], np.float32)
    sine = ad.stage(lambda s, _x: np.sin(s), 0.5, x)

    nested = ad.stage(lambda s, x: sine(s, x) * x, 0.5, x)

    assert type(sine(0.5, x)) is np.float64
    assert nested(0.5, x).dtype == (np.sin(0.5) * x).dtype == np.float64
    assert_allclose(nested(0.5, x), np.sin(0.5) * x)


# Array API functions of a Python scalar, through the namespace a NumPy array reports.
_NAMESPACE_SCALARS: dict[str, Callable[[Any, Any], Any]] = {
    "conj": lambda xp, s: xp.conj(s),
    "multiply": lambda xp, s: xp.multiply(s, s),
    "negative": lambda xp, s: xp.negative(s),
    "pow": lambda xp, s: xp.pow(s, s),
}


@pytest.mark.parametrize("scalar", list(_NAMESPACE_SCALARS.values()), ids=list(_NAMESPACE_SCALARS))
def test_array_api_functions_of_a_python_scalar_are_strong_in_every_lifetime(
    scalar: Callable[[Any, Any], Any],
) -> None:
    # Staging read an Array API function of weak scalars as Python's operator:
    # float32 where eager NumPy computes float64, and a weak conj that replay
    # could not apply to a tracer ("object has no attribute 'conjugate'").
    def program(s: Any, x32: Any, _x64: Any) -> Any:
        return scalar(x32.__array_namespace__(), s) * x32

    args = (0.5, np.array([0.5, -1.0, 2.0], np.float32), np.array([0.25, 1.5, -0.75]))
    expected = program(*args)
    staged = ad.stage(program, *args)

    assert expected.dtype == np.float64
    for lifetime, result in _lifetimes(program, args).items():
        assert result.dtype == expected.dtype, lifetime
        assert_allclose(result, expected, rtol=1e-12, err_msg=lifetime)
    gradient = ad.grad(lambda *values: np.sum(staged(*values)))(*args)
    reference = ad.grad(lambda *values: np.sum(program(*values)))(*args)
    assert type(gradient) is type(reference) is float
    assert gradient == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize("scalar", list(_NAMESPACE_SCALARS.values()), ids=list(_NAMESPACE_SCALARS))
def test_a_staged_array_api_function_of_a_python_scalar_replays_on_array_api_strict(
    scalar: Callable[[Any, Any], Any],
) -> None:
    # The reference provider's functions reject Python scalars alone, so replay
    # computes on strong arrays of them, as a NumPy function converts them.
    values = [0.3, 0.6, 0.9]

    def objective(s: Any, x: Any) -> Any:
        return scalar(x.__array_namespace__(), s) * x

    x = strict.asarray(values, dtype=strict.float32)
    staged = ad.stage(objective, 0.7, x)
    expected = objective(0.7, np.asarray(values, np.float32))
    results = (
        staged(0.7, x),
        ad.jvp(staged, argnums=(0, 1))(0.7, x, tangents=(1.0, strict.ones_like(x)))[0],
        # With the scalar held, replay multiplied a strict array by a tracer,
        # which strict rejects: "Expected Array or Python scalar".
        ad.jvp(staged, argnums=1)(0.7, x, tangents=strict.ones_like(x))[0],
        ad.stage(lambda s, x: staged(s, x), 0.7, x)(0.7, x),  # noqa: PLW0108 - trace boundary
    )

    for result in results:
        assert result.dtype == strict.float64
        assert_allclose(np.asarray(result), expected, rtol=1e-12)


# NumPy's real() and imag() return the attribute, which a Python scalar holds.
_SCALAR_PARTS: dict[str, Callable[[Any, Any], Any]] = {
    "np.real": lambda _xp, s: np.real(s),
    "np.imag": lambda _xp, s: np.imag(s),
    "xp.real": lambda xp, s: xp.real(s),
    "xp.imag": lambda xp, s: xp.imag(s),
}


@pytest.mark.parametrize("part", list(_SCALAR_PARTS.values()), ids=list(_SCALAR_PARTS))
def test_real_and_imag_parts_of_a_python_scalar_stay_weak_in_every_lifetime(
    part: Callable[[Any, Any], Any],
) -> None:
    # A traced np.real(s) recorded a strong result around a Python float, so
    # the next Python operator promoted it strongly: float64 where eager NumPy
    # computes float32.
    def program(s: Any, x32: Any, _x64: Any) -> Any:
        return (part(x32.__array_namespace__(), s) + 0.5) * x32

    args = (0.5, np.array([0.5, -1.0, 2.0], np.float32), np.array([0.25, 1.5, -0.75]))
    expected = program(*args)

    assert expected.dtype == np.float32
    for lifetime, result in _lifetimes(program, args).items():
        assert result.dtype == expected.dtype, lifetime
        assert_allclose(result, expected, err_msg=lifetime)


def _augmented(s: Any, x: Any) -> Any:
    s += 1.0
    s *= 2.0
    s += x
    return s


def test_augmented_assignment_rebinds_a_python_scalar_in_every_lifetime() -> None:
    # A Python float has no in-place operators, so `s += value` rebinds `s`;
    # a traced or staged Python scalar input was rejected as a mutated input.
    x = np.array([0.5, -1.0, 2.0], np.float32)
    expected = _augmented(0.5, x)

    value, tangent = ad.jvp(_augmented, argnums=(0, 1))(0.5, x, tangents=(1.0, np.ones_like(x)))
    staged = ad.stage(_augmented, 0.5, x)(0.5, x)

    for result in (value, staged):
        assert result.dtype == expected.dtype == np.float32
        assert_allclose(result, expected)
    assert_allclose(tangent, np.full(3, 3.0, np.float32))


_ARRAY_API_SCALARS: dict[str, Callable[[Any], Any]] = {
    "s * s": lambda s: s * s,
    "-s": operator.neg,
    "+s": operator.pos,
    "abs(s)": abs,
    "s + 1.0": lambda s: s + 1.0,
    "2.0 / s": lambda s: 2.0 / s,
    "s ** 2": lambda s: s**2,
    "s ** 2.5": lambda s: s**2.5,
    "s // 0.3": lambda s: s // 0.3,
    "s % 0.3": lambda s: s % 0.3,
    "s.real": lambda s: s.real,
    # The imag transpose asked the provider to classify a Python float, which
    # array_api_strict rejects: "'dtype' must be a dtype, not a <class 'float'>".
    "s.imag": lambda s: s.imag,
    "s.imag + s": lambda s: s.imag + s,
    "s.real * s": lambda s: s.real * s,
}


@pytest.mark.parametrize("scalar", list(_ARRAY_API_SCALARS.values()), ids=list(_ARRAY_API_SCALARS))
def test_array_api_python_operators_keep_a_selected_scalar_weak(
    scalar: Callable[[Any], Any],
) -> None:
    # Python operators on a selected Python scalar compute as Python does, which
    # the reference provider requires: its functions reject two Python scalars.
    values = [0.3, 0.6, 0.9]
    x = strict.asarray(values, dtype=strict.float32)

    def objective(s: Any, x: Any) -> Any:
        return scalar(s) * x

    def loss(s: Any, x: Any) -> Any:
        product = objective(s, x)
        return product.__array_namespace__().sum(product)

    staged = ad.stage(objective, 0.7, x)
    tangents = (1.0, strict.ones_like(x))
    results = (
        ad.jvp(objective, argnums=(0, 1))(0.7, x, tangents=tangents)[0],
        staged(0.7, x),
        ad.jvp(staged, argnums=(0, 1))(0.7, x, tangents=tangents)[0],
        ad.stage(lambda s, x: staged(s, x), 0.7, x)(0.7, x),  # noqa: PLW0108 - trace boundary
    )
    gradient = ad.grad(loss, argnums=(0, 1))(0.7, x)[0]

    for result in results:
        assert result.dtype == objective(0.7, x).dtype == strict.float32
    # The NumPy frontend differentiates the same program through the same rules.
    reference = ad.grad(loss, argnums=0)(0.7, np.asarray(values, np.float32))
    assert type(gradient) is type(reference) is float
    assert gradient == pytest.approx(reference, rel=1e-6)
    # A second derivative runs the rules on weak primals, which the reference
    # provider's functions and creation from a prototype reject as Python scalars.
    curvature = ad.grad(ad.grad(loss, argnums=0), argnums=0)
    assert curvature(0.7, x) == pytest.approx(
        curvature(0.7, np.asarray(values, np.float32)), rel=1e-6
    )


# Derivative rules at singular points of a selected Python scalar.
_SINGULAR_SCALARS: dict[str, Callable[[Any, Any], Any]] = {
    "xp.divide(1.0, s)": lambda xp, s: xp.divide(1.0, s),
    "xp.atan2(s, s)": lambda xp, s: xp.atan2(s, s),
    "s ** s": lambda _xp, s: s**s,
    "(2 * s) ** s": lambda _xp, s: (2 * s) ** s,
}


@pytest.mark.parametrize("scalar", list(_SINGULAR_SCALARS.values()), ids=list(_SINGULAR_SCALARS))
def test_array_api_rules_on_a_selected_scalar_compute_as_numpy_at_singular_points(
    scalar: Callable[[Any, Any], Any],
) -> None:
    # A rule combined weak primals with Python's operators on array_api_strict,
    # so at s = 0 its partials raised ZeroDivisionError, or left a Python float
    # where the next rule needs an array, while NumPy computes inf, nan or 0.
    values = [0.3, 0.6, 0.9]

    def loss(s: Any, x: Any) -> Any:
        product = scalar(x.__array_namespace__(), s) * x
        return product.__array_namespace__().sum(product)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = ad.grad(loss, argnums=(0, 1))(0.0, np.asarray(values, np.float32))
        gradient = ad.grad(loss, argnums=(0, 1))(0.0, strict.asarray(values, dtype=strict.float32))

    assert type(gradient[0]) is type(expected[0]) is float
    assert gradient[1].dtype == strict.float32
    assert_allclose(gradient[0], expected[0])
    assert_allclose(np.asarray(gradient[1]), expected[1])
