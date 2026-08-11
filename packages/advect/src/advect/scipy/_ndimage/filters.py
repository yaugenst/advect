# ruff: noqa: A001, A002, ANN401, PLR0913, PLR2004
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Install the Gaussian, uniform, convolution, and correlation primitives.

These registrations combine shared call normalization with the linear stencil
derivative engine.  Public SciPy-compatible signatures and concrete fast paths
remain in :mod:`advect.scipy.ndimage`.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core import primitive
from advect.scipy._frontend import _is_traced_value
from advect.scipy._ndimage.common import (
    _call_primitive,
    _cast_tangent,
    _mode_name,
    _ndim_of,
    _normalize_axes,
    _normalize_axis,
    _normalize_modes,
    _normalize_origins,
    _operand_dtype,
    _project_cotangent,
    _require_numpy_values,
    _result_spec,
    _runtime_output,
    _sample_array,
    _zero_tangent,
)
from advect.scipy._ndimage.stencil import (
    _stencil_input_transpose,
    _stencil_weight_transpose,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core import AbstractValue, ArraySpec
    from advect.core._primitive import Primitive


def _gaussian_kernel(sigma: float, order: int, radius: int) -> np.ndarray:
    if order < 0:
        raise ValueError("order must be non-negative")
    exponents = np.arange(order + 1)
    sigma_squared = sigma * sigma
    positions = np.arange(-radius, radius + 1)
    gaussian = np.exp(-0.5 / sigma_squared * positions**2)
    gaussian = gaussian / gaussian.sum()
    if order == 0:
        return gaussian
    coefficients = np.zeros(order + 1)
    coefficients[0] = 1
    derivative = np.diag(exponents[1:], 1)
    product = np.diag(np.ones(order) / -sigma_squared, -1)
    for _ in range(order):
        coefficients = (derivative + product).dot(coefficients)
    polynomial = (positions[:, None] ** exponents).dot(coefficients)
    return polynomial * gaussian


def _run_gaussian(
    input: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    sigmas: tuple[object, ...],
    orders: tuple[object, ...],
    modes: tuple[str, ...],
    truncate: float,
    radii: tuple[object, ...],
) -> Any:
    if one_dimensional:
        return _scipy_ndimage.gaussian_filter1d(
            input,
            sigmas[0],
            axis=axes[0],
            order=orders[0],
            output=output,
            mode=modes[0],
            cval=cval,
            truncate=truncate,
            radius=radii[0],
        )
    return _scipy_ndimage.gaussian_filter(
        input,
        sigmas,
        order=orders,
        output=output,
        mode=modes,
        cval=cval,
        truncate=truncate,
        radius=radii,
        axes=axes,
    )


def _install_gaussian(name: str, *, one_dimensional: bool) -> Primitive[..., Any]:
    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "sigmas",
            "orders",
            "modes",
            "truncate",
            "radii",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, cval)
        return _run_gaussian(
            input,
            cval,
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> ArraySpec:
        sample_shape = (1,) * len(input.spec.shape)
        result = _run_gaussian(
            _sample_array(input, shape=sample_shape),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, _cval = primals
        input_tangent, cval_tangent = tangents
        if input_tangent is None and cval_tangent is None:
            return np.zeros_like(output)
        result = concrete(
            input=_zero_tangent(input, input_tangent),
            cval=0 if cval_tangent is None else cval_tangent,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
            output_dtype=_operand_dtype(output).str,
        )
        return _cast_tangent(result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None]:
        del output_dtype
        input, cval = primals
        active = {0, 1} if active_input_indices is None else set(active_input_indices)
        if not active:
            return None, None
        result = cotangent
        boundary_cotangent = np.zeros((), dtype=_operand_dtype(cotangent))
        configurations = tuple(zip(axes, sigmas, orders, modes, radii, strict=True))
        for axis, sigma, order, mode, radius in reversed(configurations):
            sigma_value = float(cast("Any", sigma))
            if not one_dimensional and sigma_value <= 1e-15:
                continue
            radius_value = (
                int(truncate * sigma_value + 0.5)
                if radius is None
                else operator.index(cast("Any", radius))
            )
            result, boundary = _stencil_input_transpose(
                result,
                _gaussian_kernel(
                    sigma_value,
                    operator.index(cast("Any", order)),
                    radius_value,
                )[::-1],
                axes=(axis,),
                origins=(0,),
                modes=(mode,),
                convolution=False,
            )
            boundary_cotangent = boundary_cotangent + boundary
        return (
            _project_cotangent(result, input, output) if 0 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 1 in active else None,
        )

    return concrete


_gaussian_filter_primitive = _install_gaussian("gaussian_filter", one_dimensional=False)
_gaussian_filter1d_primitive = _install_gaussian("gaussian_filter1d", one_dimensional=True)


def _run_uniform(
    input: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    sizes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
) -> Any:
    if one_dimensional:
        return _scipy_ndimage.uniform_filter1d(
            input,
            sizes[0],
            axis=axes[0],
            output=output,
            mode=modes[0],
            cval=cval,
            origin=origins[0],
        )
    return _scipy_ndimage.uniform_filter(
        input,
        size=sizes,
        output=output,
        mode=modes,
        cval=cval,
        origin=origins,
        axes=axes,
    )


def _install_uniform(name: str, *, one_dimensional: bool) -> Primitive[..., Any]:
    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "sizes",
            "origins",
            "modes",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, cval)
        return _run_uniform(
            input,
            cval=cval,
            output=_runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> ArraySpec:
        result = _run_uniform(
            _sample_array(input, shape=(1,) * len(input.spec.shape)),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, _cval = primals
        input_tangent, cval_tangent = tangents
        if input_tangent is None and cval_tangent is None:
            return np.zeros_like(output)
        result = concrete(
            input=_zero_tangent(input, input_tangent),
            cval=0 if cval_tangent is None else cval_tangent,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
            output_dtype=_operand_dtype(output).str,
        )
        return _cast_tangent(result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None]:
        del output_dtype
        input, cval = primals
        active = {0, 1} if active_input_indices is None else set(active_input_indices)
        if not active:
            return None, None
        symmetric = all(
            size % 2 == 1 and origin == 0 and _mode_name(mode) in {"constant", "reflect", "wrap"}
            for size, origin, mode in zip(sizes, origins, modes, strict=True)
        )
        if active == {0} and symmetric:
            arguments = {
                "axes": axes,
                "sizes": sizes,
                "origins": origins,
                "modes": modes,
            }
            result = (
                concrete(
                    input=cotangent,
                    cval=0,
                    output_dtype=_operand_dtype(cotangent).str,
                    **arguments,
                )
                if _is_traced_value(cotangent)
                else _run_uniform(
                    cotangent,
                    0,
                    None,
                    one_dimensional=one_dimensional,
                    **arguments,
                )
            )
            return _project_cotangent(result, input, output), None
        result = cotangent
        boundary_cotangent = np.zeros((), dtype=_operand_dtype(cotangent))
        configurations = tuple(zip(axes, sizes, origins, modes, strict=True))
        for axis, size, origin, mode in reversed(configurations):
            if not one_dimensional and size <= 1:
                continue
            result, boundary = _stencil_input_transpose(
                result,
                np.full(size, 1.0 / size),
                axes=(axis,),
                origins=(origin,),
                modes=(mode,),
                convolution=False,
            )
            boundary_cotangent = boundary_cotangent + boundary
        return (
            _project_cotangent(result, input, output) if 0 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 1 in active else None,
        )

    return concrete


_uniform_filter_primitive = _install_uniform("uniform_filter", one_dimensional=False)
_uniform_filter1d_primitive = _install_uniform("uniform_filter1d", one_dimensional=True)


def _run_correlation(
    function: Callable[..., Any],
    input: Any,
    weights: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    mode_sequence: bool,
) -> Any:
    runtime_mode: object = modes if mode_sequence else (modes[0] if modes else "reflect")
    if one_dimensional:
        return function(
            input,
            weights,
            axis=axes[0],
            output=output,
            mode=runtime_mode,
            cval=cval,
            origin=origins[0],
        )
    return function(
        input,
        weights,
        output=output,
        mode=runtime_mode,
        cval=cval,
        origin=origins,
        axes=axes,
    )


def _correlation_kernel_configuration(
    input: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    kernel_axes = tuple(sorted(axes))
    if len(axes) == input.ndim:
        return kernel_axes, origins, modes
    origins_by_axis = dict(zip(axes, origins, strict=True))
    modes_by_axis = dict(zip(axes, modes, strict=True))
    return (
        kernel_axes,
        tuple(origins_by_axis[axis] for axis in kernel_axes),
        tuple(modes_by_axis[axis] for axis in kernel_axes),
    )


def _install_correlation(
    name: str,
    *,
    one_dimensional: bool,
    convolution: bool,
) -> Primitive[..., Any]:
    function = cast("Callable[..., Any]", getattr(_scipy_ndimage, name))

    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "origins",
            "modes",
            "mode_sequence",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        weights: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, weights, cval)
        return _run_correlation(
            function,
            input,
            weights,
            cval,
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            origins=origins,
            modes=modes,
            mode_sequence=mode_sequence,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        weights: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> ArraySpec:
        result = _run_correlation(
            function,
            _sample_array(input, shape=(1,) * len(input.spec.shape)),
            _sample_array(weights),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            origins=origins,
            modes=modes,
            mode_sequence=mode_sequence,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, weights, cval = primals
        input_tangent, weights_tangent, cval_tangent = tangents
        result: Any | None = None
        if input_tangent is not None or cval_tangent is not None:
            result = concrete(
                input=_zero_tangent(input, input_tangent),
                weights=weights,
                cval=0 if cval_tangent is None else cval_tangent,
                axes=axes,
                origins=origins,
                modes=modes,
                mode_sequence=mode_sequence,
                output_dtype=_operand_dtype(output).str,
            )
        if weights_tangent is not None:
            weight_term = concrete(
                input=input,
                weights=weights_tangent,
                cval=cval,
                axes=axes,
                origins=origins,
                modes=modes,
                mode_sequence=mode_sequence,
                output_dtype=_operand_dtype(output).str,
            )
            result = weight_term if result is None else result + weight_term
        return _cast_tangent(np.zeros_like(output) if result is None else result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None, Any | None]:
        del mode_sequence, output_dtype
        input, weights, cval = primals
        active = {0, 1, 2} if active_input_indices is None else set(active_input_indices)
        kernel_axes, kernel_origins, kernel_modes = _correlation_kernel_configuration(
            input,
            axes=axes,
            origins=origins,
            modes=modes,
        )
        input_cotangent: Any | None = None
        boundary_cotangent: Any | None = None
        if active.intersection({0, 2}):
            input_cotangent, boundary_cotangent = _stencil_input_transpose(
                cotangent,
                weights,
                axes=kernel_axes,
                origins=kernel_origins,
                modes=kernel_modes,
                convolution=convolution,
            )
        weight_cotangent = (
            _stencil_weight_transpose(
                cotangent,
                input,
                weights,
                cval,
                axes=kernel_axes,
                origins=kernel_origins,
                modes=kernel_modes,
                convolution=convolution,
            )
            if 1 in active
            else None
        )
        return (
            _project_cotangent(input_cotangent, input, output) if 0 in active else None,
            _project_cotangent(weight_cotangent, weights, output) if 1 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 2 in active else None,
        )

    return concrete


_convolve_primitive = _install_correlation(
    "convolve",
    one_dimensional=False,
    convolution=True,
)
_correlate_primitive = _install_correlation(
    "correlate",
    one_dimensional=False,
    convolution=False,
)
_convolve1d_primitive = _install_correlation(
    "convolve1d",
    one_dimensional=True,
    convolution=True,
)
_correlate1d_primitive = _install_correlation(
    "correlate1d",
    one_dimensional=True,
    convolution=False,
)


def _correlation_call(  # noqa: PLR0917 - mirrors the SciPy call signature
    name: str,
    primitive_function: Primitive[..., Any],
    input: object,
    weights: object,
    output: object,
    mode: object,
    cval: object,
    origin: object,
    axes: object,
    *,
    one_dimensional: bool,
) -> object:
    ndim = _ndim_of(input)
    normalized_axes = (
        (_normalize_axis(axes, ndim),) if one_dimensional else _normalize_axes(axes, ndim)
    )
    normalized_origins = _normalize_origins(origin, len(normalized_axes))
    mode_sequence = not isinstance(mode, str) and isinstance(mode, Iterable)
    normalized_modes = _normalize_modes(mode, len(normalized_axes))
    return _call_primitive(
        primitive_function,
        name=name,
        input=input,
        output=output,
        operands={"weights": weights, "cval": cval},
        static={
            "axes": normalized_axes,
            "origins": normalized_origins,
            "modes": normalized_modes,
            "mode_sequence": mode_sequence,
        },
    )
