# ruff: noqa: ANN401  # Primitive checks are intentionally array-provider generic.
"""Numerical validation for composed functions and custom primitives."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, NoReturn, cast

from advect.autodiff.api._scalar_boundary import _is_complex_numeric
from advect.core._abstract import AbstractValue, ArraySpec, _scalar_spec
from advect.core._array_api.providers import _get_array_namespace
from advect.core._primitive import MissingPrimitiveRuleError
from advect.core._primitive_call import _infer_namespace, _normalize_output_pytree
from advect.core._pytree import tree_flatten, tree_map, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from advect.core._primitive import Primitive


def _ones_like(value: Any) -> Any:
    namespace = _get_array_namespace(value)
    ones_like = getattr(namespace, "ones_like", None) if namespace is not None else None
    if callable(ones_like):
        return ones_like(value)
    if isinstance(value, (bool, int, float, complex)):
        return type(value)(1)
    return value * 0 + 1


def _scaled_difference(upper: Any, lower: Any, *, scale: float) -> Any:
    return (upper - lower) / scale


def _tree_difference(positive: Any, negative: Any, scale: float) -> Any:
    return tree_map(partial(_scaled_difference, scale=scale), positive, negative)


def _shift_leaf(value: Any, step: Any, *, scale: float) -> Any:
    return value + scale * step


def _as_abstract_value(value: Any) -> AbstractValue:
    return AbstractValue(_scalar_spec(value))


def _tree_allclose(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    actual_leaves, actual_treedef = tree_flatten(actual)
    expected_leaves, expected_treedef = tree_flatten(expected)
    if actual_treedef != expected_treedef:
        return False
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        namespace = _get_array_namespace(expected_leaf) or _get_array_namespace(actual_leaf)
        difference = abs(actual_leaf - expected_leaf)
        tolerance = atol + rtol * abs(expected_leaf)
        comparison = difference <= tolerance
        all_function = getattr(namespace, "all", None) if namespace is not None else None
        if callable(all_function):
            comparison = all_function(comparison)
        else:
            all_method = getattr(comparison, "all", None)
            if callable(all_method):
                comparison = all_method()
        if not bool(comparison):
            return False
    return True


def _copy_check_value(value: Any) -> Any:
    copy_method = getattr(value, "copy", None)
    if callable(copy_method):
        return copy_method()
    return value


def _check_same_tree_specs(
    primitive: Primitive[Any, Any],
    actual: Any,
    expected: Any,
    *,
    phase: str,
) -> None:
    actual_leaves, actual_treedef = tree_flatten(actual)
    expected_leaves, expected_treedef = tree_flatten(expected)
    if actual_treedef != expected_treedef:
        msg = f"Primitive '{primitive.name}' {phase} output structure differs from concrete output"
        raise AssertionError(msg)
    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True),
    ):
        actual_spec = _scalar_spec(actual_leaf)
        expected_spec = _scalar_spec(expected_leaf)
        if actual_spec.shape != expected_spec.shape or str(actual_spec.dtype) != str(
            expected_spec.dtype
        ):
            msg = (
                f"Primitive '{primitive.name}' {phase} output leaf {index} disagrees "
                f"with concrete metadata: expected shape={expected_spec.shape}, "
                f"dtype={expected_spec.dtype}; got shape={actual_spec.shape}, "
                f"dtype={actual_spec.dtype}"
            )
            raise AssertionError(msg)


def _check_tree_unchanged(
    primitive: Primitive[Any, Any],
    actual: Any,
    snapshot: Any,
    *,
    phase: str,
) -> None:
    _check_same_tree_specs(primitive, actual, snapshot, phase=f"{phase} input")
    if not _tree_allclose(actual, snapshot, atol=0.0, rtol=0.0):
        msg = f"Primitive '{primitive.name}' {phase} mutated an input"
        raise AssertionError(msg)


def _real_inner_product(left: Sequence[Any], right: Sequence[Any]) -> float:
    total = 0.0
    for left_leaf, right_leaf in zip(left, right, strict=True):
        if left_leaf is None or right_leaf is None:
            continue
        namespace = _get_array_namespace(left_leaf) or _get_array_namespace(right_leaf)
        conjugate = getattr(namespace, "conj", None) if namespace is not None else None
        if callable(conjugate):
            product = conjugate(left_leaf) * right_leaf
        else:
            conjugate_method = getattr(left_leaf, "conjugate", None)
            product = (
                conjugate_method() * right_leaf
                if callable(conjugate_method)
                else left_leaf * right_leaf
            )
        sum_function = getattr(namespace, "sum", None) if namespace is not None else None
        value = sum_function(product) if callable(sum_function) else product
        sum_method = getattr(value, "sum", None)
        if not callable(sum_function) and callable(sum_method):
            value = sum_method()
        total += float(cast("Any", getattr(value, "real", value)))
    return total


def _check_output_specs(primitive: Primitive[Any, Any], concrete: Any, abstract: Any) -> None:
    concrete_leaves, concrete_treedef = tree_flatten(concrete)
    abstract_leaves, abstract_treedef = tree_flatten(abstract)
    if concrete_treedef != abstract_treedef:
        msg = f"Primitive '{primitive.name}' abstract output structure differs from concrete output"
        raise AssertionError(msg)
    for index, (value, abstract_value) in enumerate(
        zip(concrete_leaves, abstract_leaves, strict=True)
    ):
        spec = abstract_value.spec if isinstance(abstract_value, AbstractValue) else abstract_value
        if not isinstance(spec, ArraySpec):
            msg = f"Primitive '{primitive.name}' abstract output leaf {index} is not an ArraySpec"
            raise AssertionError(msg)  # noqa: TRY004 - this is a failed author check.
        concrete_spec = _scalar_spec(value)
        if concrete_spec.shape != spec.shape or str(concrete_spec.dtype) != str(spec.dtype):
            msg = (
                f"Primitive '{primitive.name}' abstract output leaf {index} disagrees with "
                f"concrete output: expected shape={concrete_spec.shape}, "
                f"dtype={concrete_spec.dtype}; got shape={spec.shape}, dtype={spec.dtype}"
            )
            raise AssertionError(msg)


def _default_check_tangent(value: Any, *, complex_direction: bool) -> Any:
    tangent = _ones_like(value)
    if complex_direction and _is_complex_numeric(value):
        return tangent * (1 + 1j)
    return tangent


def _missing_rule(
    primitive: Primitive[Any, Any],
    capability: str,
    decorator: str,
) -> NoReturn:
    msg = f"Primitive '{primitive.name}' is missing {capability!r}; define it with {decorator}."
    raise MissingPrimitiveRuleError(msg)


def _custom_primitives_for_call(
    function: Callable[..., Any],
    *,
    primal: Any,
) -> tuple[str, ...]:
    from advect.autodiff._ephemeral import trace_call  # noqa: PLC0415

    trace = trace_call(
        function,
        args=(primal,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        return tuple(
            op.removeprefix("custom.") for op in trace.tape.op_names if op.startswith("custom.")
        )
    finally:
        trace.tape.release_payloads()


def check_gradient(
    function: Callable[..., Any],
    primal: Any,
    *,
    tangent: Any | None = None,
    epsilons: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5),
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Check a unary composed function against directional differences.

    The check compares Advect's whole-function JVP with a central finite-
    difference sweep, then checks the reverse gradient against the same
    directional derivative. It checks consistency with the function that
    actually ran, not whether that function encodes the intended mathematics.

    Parameters
    ----------
    function
        Unary function with a real scalar output. This may be a composition of
        built-in operations and public custom primitives.
    primal
        Representative input value or pytree at which to check `function`.
    tangent
        Direction pytree matching `primal`. When omitted, every numeric leaf
        receives an all-ones direction.
    epsilons
        Non-empty sequence of finite positive central-difference steps. The
        JVP comparison passes when at least one step agrees within tolerance.
    atol
        Absolute tolerance for the finite-difference and real-adjoint checks.
    rtol
        Relative tolerance for the finite-difference and real-adjoint checks.

    Returns
    -------
    None
        The check returns only when both comparisons pass.

    Raises
    ------
    ValueError
        If `epsilons` is empty or contains a non-finite or non-positive step,
        or if the tangent, input, or scalar-output structure is invalid.
    TypeError
        If a selected input or tangent leaf is unsupported by the active array
        provider.
    NoJVPError
        If an operation on the checked path has no forward-mode rule.
    NoVJPError
        If an operation on the checked path has no reverse-mode rule.
    AssertionError
        If the JVP disagrees with every finite-difference step or the reverse
        gradient violates the JVP's real-adjoint identity. The error names any
        custom primitives observed on the failing path.

    Notes
    -----
    This is a representative author check, not exhaustive conformance
    evidence. Run it on a composed public path in addition to testing each
    custom primitive with `check_primitive`.
    """
    steps = tuple(epsilons)
    if not steps or any(not math.isfinite(step) or step <= 0 for step in steps):
        msg = "check_gradient epsilons must be a non-empty sequence of positive values"
        raise ValueError(msg)

    direction = tree_map(_ones_like, primal) if tangent is None else tangent
    tangent_leaves = tree_flatten(direction)[0]

    from advect.autodiff.api.forward import jvp  # noqa: PLC0415
    from advect.autodiff.api.reverse import grad  # noqa: PLC0415

    _output, directional = jvp(function)(primal, tangents=direction)

    finite_difference_passed = False
    for epsilon in steps:
        positive = tree_map(
            partial(_shift_leaf, scale=epsilon),
            primal,
            direction,
        )
        negative = tree_map(
            partial(_shift_leaf, scale=-epsilon),
            primal,
            direction,
        )
        finite_difference = _tree_difference(
            function(positive),
            function(negative),
            2 * epsilon,
        )
        if _tree_allclose(directional, finite_difference, atol=atol, rtol=rtol):
            finite_difference_passed = True

    directional_leaves = tree_flatten(directional)[0]
    gradient_leaves = tree_flatten(grad(function)(primal))[0]
    left = _real_inner_product([1.0], directional_leaves)
    right = _real_inner_product(gradient_leaves, tangent_leaves)
    adjoint_error = abs(left - right)
    adjoint_passed = adjoint_error <= atol + rtol * abs(left)
    failures: list[str] = []
    if not finite_difference_passed:
        failures.append(
            f"the JVP disagreed with central finite differences at every epsilon {steps!r}"
        )
    if not adjoint_passed:
        failures.append(
            "the reverse gradient disagreed with the JVP "
            f"(left={left}, right={right}, error={adjoint_error})"
        )
    if failures:
        custom_primitives = _custom_primitives_for_call(function, primal=primal)
        primitive_hint = (
            " Custom primitives on this path: "
            f"{', '.join(custom_primitives)}; run check_primitive on each authoring handle."
            if custom_primitives
            else ""
        )
        name = getattr(function, "__qualname__", type(function).__name__)
        msg = f"Gradient check for {name!r} failed: {'; '.join(failures)}.{primitive_hint}"
        raise AssertionError(msg)


def check_primitive(  # noqa: C901, PLR0912, PLR0913, PLR0915
    primitive: Primitive[Any, Any],
    *,
    primals: tuple[Any, ...],
    static: Mapping[str, Any] | None = None,
    tangents: tuple[Any, ...] | None = None,
    cotangent: Any | None = None,
    check: tuple[str, ...] = ("abstract", "jvp", "transpose"),
    epsilon: float = 1e-4,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Run selected author checks for one representative primitive invocation.

    The default ``("abstract", "jvp", "transpose")`` is a first-order smoke
    check. It does not stage the primitive or check input preservation. Authors
    of a serializable non-residual primitive should normally run
    ``("abstract", "jvp", "transpose", "nested", "stage")`` for every
    materially different shape, dtype, and static-argument form. Add
    ``"complex"`` in a separate call whose primals are complex when the
    primitive supports complex values. Residual primitives are first-order
    boundaries and therefore omit ``"nested"``. A transpose-only primitive may
    request just ``"transpose"``; the check then compares its explicit rule
    with a central finite difference. The ``"jvp"``, ``"complex"``, and
    ``"nested"`` checks require a JVP.

    The stage check executes both the compiled and serialized program, compares
    output structure, shape, and dtype exactly, and verifies that inputs remain
    unchanged. Repository-wide support still requires the conformance inventory;
    this helper intentionally does not import Hypothesis or claim exhaustive
    coverage from one sample.

    Parameters
    ----------
    primitive
        Authoring handle returned by ``advect.primitive``.
    primals
        One representative value for each non-static implementation argument,
        in implementation-parameter order. Nested pytrees are preserved.
    static
        Values for declared static arguments that do not use their
        implementation defaults.
    tangents
        Optional tangent pytree matching ``primals``. Ones are used by
        default; leaves of nondifferentiable arguments are ignored.
    cotangent
        Optional pytree matching the primitive output. Ones are used by
        default.
    check
        Any of ``"abstract"``, ``"jvp"``, ``"transpose"``, ``"complex"``,
        ``"nested"``, and ``"stage"``. Only ``"stage"`` compiles and restores
        a program and checks that those executions preserve their inputs.
    epsilon
        Central finite-difference step.
    atol
        Absolute numerical tolerance.
    rtol
        Relative numerical tolerance.

    Raises
    ------
    ValueError
        If the requested checks or representative inputs are invalid.
    MissingPrimitiveRuleError
        If a requested capability has no required author rule.
    AssertionError
        If an installed rule, staged execution, or numerical identity fails
        its check.
    """
    requested = tuple(check)
    supported = {"abstract", "jvp", "transpose", "complex", "nested", "stage"}
    unknown = set(requested).difference(supported)
    if unknown:
        joined = ", ".join(sorted(unknown))
        msg = f"Unknown primitive check(s): {joined}"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)

    dynamic_names = primitive._dynamic_argnames  # noqa: SLF001
    if len(primals) != len(dynamic_names):
        msg = (
            f"Primitive '{primitive.name}' check expected {len(dynamic_names)} dynamic "
            f"primals ({', '.join(dynamic_names)}), got {len(primals)}"
        )
        raise ValueError(msg)
    call_arguments = dict(zip(dynamic_names, primals, strict=True))
    call_arguments.update({} if static is None else static)
    bound = primitive._bind_call((), call_arguments)  # noqa: SLF001
    static_arguments = {name: bound[name] for name in primitive.static_argnames}

    def invoke(values: tuple[Any, ...]) -> Any:
        arguments = dict(zip(dynamic_names, values, strict=True))
        arguments.update(static_arguments)
        return primitive(**arguments)

    def dynamic_call(*values: Any) -> Any:
        return invoke(values)

    concrete = invoke(primals)
    input_leaves, _input_treedef = tree_flatten(primals)
    rule_namespace = _infer_namespace(input_leaves)

    def normalize_rule_output(value: Any) -> Any:
        leaves, treedef = _normalize_output_pytree(
            value,
            namespace=rule_namespace,
        )
        return tree_unflatten(treedef, leaves)

    rule_concrete = normalize_rule_output(concrete)
    if "abstract" in requested:
        abstract_rule = primitive._abstract_rule  # noqa: SLF001
        if abstract_rule is None:
            _missing_rule(primitive, "abstract", "@primitive.def_abstract")
        abstract_arguments = {
            name: tree_map(_as_abstract_value, primal)
            for name, primal in zip(dynamic_names, primals, strict=True)
        }
        abstract_arguments.update(static_arguments)
        abstract = abstract_rule(**abstract_arguments)
        _check_output_specs(primitive, concrete, abstract)

    primal_leaves, primal_treedef = tree_flatten(primals)
    if "complex" in requested and not any(_is_complex_numeric(value) for value in primal_leaves):
        msg = f"Primitive '{primitive.name}' complex check requires at least one complex primal"
        raise ValueError(msg)

    derivative_checks = {"jvp", "transpose", "complex", "nested"}
    needs_derivative = bool(derivative_checks.intersection(requested))
    if not needs_derivative:
        if "stage" in requested:
            _check_primitive_stage(
                primitive,
                primals=primals,
                concrete=concrete,
                dynamic_call=dynamic_call,
                atol=atol,
                rtol=rtol,
            )
        return

    jvp_rule = primitive._jvp_rule  # noqa: SLF001
    requires_jvp = bool({"jvp", "complex", "nested"}.intersection(requested))
    if requires_jvp and jvp_rule is None:
        _missing_rule(primitive, "jvp", "@primitive.def_jvp")

    nondiff_top_level = set(primitive.nondiff_argnames)
    nondiff_mask: list[bool] = []
    for name, primal in zip(dynamic_names, primals, strict=True):
        leaves, _treedef = tree_flatten(primal)
        nondiff_mask.extend([name in nondiff_top_level] * len(leaves))

    tangent_tree = (
        tree_map(
            partial(_default_check_tangent, complex_direction="complex" in requested),
            primals,
        )
        if tangents is None
        else tangents
    )
    tangent_leaves, tangent_treedef = tree_flatten(tangent_tree)
    if tangent_treedef != primal_treedef:
        msg = "Primitive-check tangents must match the primals pytree"
        raise ValueError(msg)
    active_tangents = tuple(
        None if nondiff else tangent
        for nondiff, tangent in zip(nondiff_mask, tangent_leaves, strict=True)
    )

    def finite_difference_directional() -> Any:
        positive_leaves = [
            primal if tangent is None else primal + epsilon * tangent
            for primal, tangent in zip(primal_leaves, active_tangents, strict=True)
        ]
        negative_leaves = [
            primal if tangent is None else primal - epsilon * tangent
            for primal, tangent in zip(primal_leaves, active_tangents, strict=True)
        ]
        positive = invoke(cast("tuple[Any, ...]", tree_unflatten(primal_treedef, positive_leaves)))
        negative = invoke(cast("tuple[Any, ...]", tree_unflatten(primal_treedef, negative_leaves)))
        return _tree_difference(positive, negative, 2 * epsilon)

    jvp_result = (
        jvp_rule(
            rule_concrete,
            tuple(primal_leaves),
            active_tangents,
            **static_arguments,
        )
        if jvp_rule is not None
        else None
    )

    if {"jvp", "complex"}.intersection(requested):
        finite_difference = finite_difference_directional()
        if not _tree_allclose(jvp_result, finite_difference, atol=atol, rtol=rtol):
            msg = f"Primitive '{primitive.name}' JVP disagrees with directional finite differences"
            raise AssertionError(msg)

    needs_transpose = bool({"transpose", "complex", "nested"}.intersection(requested))
    output_cotangent = tree_map(_ones_like, rule_concrete) if cotangent is None else cotangent
    if needs_transpose:
        if primitive.has_residual and "nested" in requested:
            msg = (
                f"Primitive '{primitive.name}' uses an opaque residual and supports "
                "first-order differentiation only"
            )
            raise MissingPrimitiveRuleError(msg)
        output_cotangent_leaves, output_cotangent_treedef = tree_flatten(output_cotangent)
        _output_leaves, output_treedef = tree_flatten(rule_concrete)
        if output_cotangent_treedef != output_treedef:
            msg = "Primitive-check cotangent must match the output pytree"
            raise ValueError(msg)
        if primitive._transpose_rule is None:  # noqa: SLF001
            if jvp_rule is None:
                _missing_rule(primitive, "transpose", "@primitive.def_transpose")
            if primitive.has_residual:
                msg = (
                    f"Primitive '{primitive.name}' uses an opaque residual and requires "
                    "an explicit @primitive.def_transpose rule"
                )
                raise MissingPrimitiveRuleError(msg)
        from advect.autodiff.api.reverse import vjp  # noqa: PLC0415

        argnums = tuple(range(len(primals)))
        _value, pullback = vjp(dynamic_call, argnums=argnums)(*primals)
        contributions = pullback(output_cotangent)
        contribution_leaves, _contribution_treedef = tree_flatten(contributions)

        directional = jvp_result if jvp_rule is not None else finite_difference_directional()
        directional_leaves, directional_treedef = tree_flatten(directional)
        if directional_treedef != output_treedef:
            source = "JVP" if jvp_rule is not None else "finite-difference directional"
            msg = (
                f"Primitive '{primitive.name}' {source} output structure differs "
                "from concrete output"
            )
            raise AssertionError(msg)

        left = _real_inner_product(output_cotangent_leaves, directional_leaves)
        right = _real_inner_product(contribution_leaves, tangent_leaves)
        if abs(left - right) > atol + rtol * abs(left):
            msg = (
                f"Primitive '{primitive.name}' transpose violates the real-adjoint identity: "
                f"left={left}, right={right}"
            )
            raise AssertionError(msg)

    if "nested" in requested:
        _check_primitive_nested(
            primitive,
            primals=primals,
            tangent_tree=cast("tuple[Any, ...]", tangent_tree),
            dynamic_call=dynamic_call,
            output_cotangent=output_cotangent,
            static_arguments=static_arguments,
        )

    if "stage" in requested:
        _check_primitive_stage(
            primitive,
            primals=primals,
            concrete=concrete,
            dynamic_call=dynamic_call,
            atol=atol,
            rtol=rtol,
        )


def _check_primitive_nested(
    primitive: Primitive[Any, Any],
    *,
    primals: tuple[Any, ...],
    tangent_tree: tuple[Any, ...],
    dynamic_call: Callable[..., Any],
    output_cotangent: Any,
    static_arguments: Mapping[str, Any],
) -> None:
    from advect.autodiff.api.forward import jvp  # noqa: PLC0415

    argnums = tuple(range(len(primals)))

    def first_directional(*values: Any) -> Any:
        _value, directional = jvp(dynamic_call, argnums=argnums)(
            *values,
            tangents=tangent_tree,
        )
        return directional

    try:
        jvp(first_directional, argnums=argnums)(
            *primals,
            tangents=tangent_tree,
        )
    except Exception as error:
        msg = f"Primitive '{primitive.name}' JVP rule failed nested differentiation"
        raise AssertionError(msg) from error

    transpose_rule = primitive._transpose_rule  # noqa: SLF001
    if transpose_rule is not None:

        def transpose_application(*values: Any) -> Any:
            primal_leaves, _treedef = tree_flatten(values)
            output = dynamic_call(*values)
            return transpose_rule(
                output_cotangent,
                tuple(primal_leaves),
                output,
                **static_arguments,
            )

        try:
            jvp(transpose_application, argnums=argnums)(
                *primals,
                tangents=tangent_tree,
            )
        except Exception as error:
            msg = f"Primitive '{primitive.name}' transpose rule failed nested tracing"
            raise AssertionError(msg) from error


def _check_primitive_stage(
    primitive: Primitive[Any, Any],
    *,
    primals: tuple[Any, ...],
    concrete: Any,
    dynamic_call: Callable[..., Any],
    atol: float,
    rtol: float,
) -> None:
    from advect.core._stage import StagedProgram, stage  # noqa: PLC0415

    specs = cast("tuple[Any, ...]", tree_map(_scalar_spec, primals))
    program = cast("StagedProgram", stage(dynamic_call, specs=specs))
    restored = StagedProgram.from_dict(program.to_dict())
    snapshot = tree_map(_copy_check_value, primals)
    for phase, staged_program in (
        ("compiled stage", program),
        ("serialized stage", restored),
    ):
        staged_result = staged_program(*primals)
        _check_same_tree_specs(primitive, staged_result, concrete, phase=phase)
        if not _tree_allclose(staged_result, concrete, atol=atol, rtol=rtol):
            msg = f"Primitive '{primitive.name}' {phase} disagrees with concrete execution"
            raise AssertionError(msg)
        _check_tree_unchanged(primitive, primals, snapshot, phase=phase)


__all__ = ["check_gradient", "check_primitive"]
