"""Concrete NumPy evaluator routing helpers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from advect.numpy._op_bindings import decanonicalize_array_op

type Evaluator = Callable[[tuple[object, ...], dict[str, object]], object]
type BoundEvaluator = Callable[[tuple[object, ...]], object]
type BackendFuncInfo = tuple[Callable[..., object], frozenset[str], bool]


@dataclass(slots=True)
class ArrayProtocolEvalRuntime:
    """NumPy evaluator runtime."""

    special_evaluators: dict[str, Evaluator] = field(default_factory=dict)
    func_cache: dict[str, BackendFuncInfo] = field(default_factory=dict)

    def register_evaluator(self, op: str, evaluator: Evaluator) -> None:
        self.special_evaluators[op] = evaluator

    def _coerce_eval_result(self, value: object) -> object:
        if callable(getattr(value, "_advect_snapshot", None)):
            return value
        return np.asarray(value)

    def _get_backend_func(self, op: str) -> BackendFuncInfo | None:
        if op in self.func_cache:
            return self.func_cache[op]

        op_legacy = decanonicalize_array_op(op)
        backend_prefix = "numpy."
        if not op_legacy.startswith(backend_prefix):
            return None

        func_path = op_legacy.removeprefix(backend_prefix)
        func: object | None = np
        for part in func_path.split("."):
            func = getattr(func, part, None) if func is not None else None
        if func is None or not callable(func):
            return None

        try:
            sig = inspect.signature(func)
            valid_params = frozenset(
                name
                for name, param in sig.parameters.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            )
            accepts_var_keyword = any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )
        except (ValueError, TypeError):
            valid_params = frozenset()
            # Fallback to passing non-internal attrs when signature inspection
            # is unavailable (e.g., C-level callables without inspect metadata).
            accepts_var_keyword = True

        result = (func, valid_params, accepts_var_keyword)
        self.func_cache[op] = result
        return result

    @staticmethod
    def _filter_attrs(
        attrs: dict[str, object],
        valid_params: frozenset[str],
        *,
        accepts_var_keyword: bool,
    ) -> dict[str, object]:
        if accepts_var_keyword:
            return {k: v for k, v in attrs.items() if not k.startswith("_advect_")}
        return {k: v for k, v in attrs.items() if k in valid_params}

    @staticmethod
    def _looks_like_ufunc(func: object) -> bool:
        return callable(func) and hasattr(func, "nin") and hasattr(func, "nout")

    def evaluate_op(
        self,
        op: str,
        inputs: tuple[object, ...],
        attrs: dict[str, object],
    ) -> object:
        op_legacy = decanonicalize_array_op(op)
        if op in self.special_evaluators:
            return self.special_evaluators[op](inputs, attrs)
        if op_legacy in self.special_evaluators:
            return self.special_evaluators[op_legacy](inputs, attrs)

        func_info = self._get_backend_func(op)
        if func_info is None:
            msg = f"Unknown operation: {op}"
            raise ValueError(msg)

        func, valid_params, accepts_var_keyword = func_info
        if self._looks_like_ufunc(func):
            allowed_ufunc_kwargs = {
                "casting",
                "dtype",
                "order",
                "signature",
                "subok",
                "where",
            }
            user_kwargs = {
                k: v
                for k, v in attrs.items()
                if (not k.startswith("_advect_")) and (k in allowed_ufunc_kwargs)
            }

            return func(*inputs, **user_kwargs)

        filtered_attrs = self._filter_attrs(
            attrs,
            valid_params,
            accepts_var_keyword=accepts_var_keyword,
        )
        return func(*inputs, **filtered_attrs)

    def bind_evaluator(
        self,
        op: str,
        attrs: dict[str, object],
    ) -> BoundEvaluator | None:
        """Bind backend routing and attribute filtering for one graph node."""
        op_legacy = decanonicalize_array_op(op)
        special = self.special_evaluators.get(op)
        if special is None:
            special = self.special_evaluators.get(op_legacy)
        if special is not None:
            bound_attrs = dict(attrs)
            return lambda inputs: special(inputs, bound_attrs)

        func_info = self._get_backend_func(op)
        if func_info is None:
            return None

        func, valid_params, accepts_var_keyword = func_info
        if self._looks_like_ufunc(func):
            allowed_ufunc_kwargs = {
                "casting",
                "dtype",
                "order",
                "signature",
                "subok",
                "where",
            }
            base_kwargs = {
                key: value
                for key, value in attrs.items()
                if not key.startswith("_advect_") and key in allowed_ufunc_kwargs
            }

            if not base_kwargs:
                return lambda inputs: func(*inputs)
            return lambda inputs: func(*inputs, **base_kwargs)

        filtered_attrs = self._filter_attrs(
            attrs,
            valid_params,
            accepts_var_keyword=accepts_var_keyword,
        )
        return lambda inputs: func(*inputs, **filtered_attrs)


NUMPY_EVAL_RUNTIME = ArrayProtocolEvalRuntime()
