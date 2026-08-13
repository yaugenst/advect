"""Typed Python surface of the native tape and staged-runtime adapter."""

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, final

__version__: str
__build_profile__: Literal["debug", "release"]
__all__ = [
    "DynamicTape",
    "GraphBuilder",
    "GraphExecutionPlan",
    "GraphNode",
    "GraphStore",
    "__build_profile__",
    "__version__",
    "build_graph_execution_plan",
    "deserialize_graph_json",
    "dynamic_jvp",
    "dynamic_jvp_many",
    "dynamic_vjp",
    "dynamic_vjp_many",
    "execute_graph",
]

def deserialize_graph_json(encoded: str) -> GraphStore: ...
def build_graph_execution_plan(
    store: GraphStore,
    binder: Callable[[str, dict[str, Any]], Callable[..., object]],
) -> GraphExecutionPlan: ...
def execute_graph(
    plan: GraphExecutionPlan,
    inputs: Sequence[object],
    constants: Sequence[object],
    context: object | None = None,
) -> list[object]: ...
def dynamic_jvp(
    tape: DynamicTape,
    tangent_seeds: Sequence[tuple[int, object]],
    requested_outputs: Sequence[int],
    *,
    consume: bool = False,
) -> list[object | None]: ...
def dynamic_jvp_many(
    tape: DynamicTape,
    tangent_seed_sets: Sequence[Sequence[tuple[int, object]]],
    requested_outputs: Sequence[int],
) -> list[list[object | None]]: ...
def dynamic_vjp(
    tape: DynamicTape,
    output_cotangents: Sequence[tuple[int, object]],
    requested_inputs: Sequence[int],
    *,
    consume: bool = False,
) -> list[object | None]: ...
def dynamic_vjp_many(
    tape: DynamicTape,
    output_cotangent_sets: Sequence[Sequence[tuple[int, object]]],
    requested_inputs: Sequence[int],
) -> list[list[object | None]]: ...

@final
class DynamicTape:
    def __new__(cls) -> Self: ...
    @property
    def node_count(self) -> int: ...
    @property
    def inputs(self) -> list[int]: ...
    @property
    def op_names(self) -> list[str]: ...
    @property
    def is_consumed(self) -> bool: ...
    def record_input(
        self,
        value: object,
        shape: Sequence[int],
        dtype: object,
        *,
        name: str | None = None,
        active: bool = True,
    ) -> int: ...
    def record_operation(
        self,
        op: str,
        inputs: Sequence[int],
        value: object,
        attrs: dict[str, Any],
        shape: Sequence[int],
        dtype: object,
        *,
        schema_version: int = 1,
        name: str | None = None,
        source_location: str | None = None,
    ) -> int: ...
    def record_operation_with_literals(
        self,
        op: str,
        inputs: Sequence[int],
        input_positions: Sequence[int],
        literals: Sequence[object],
        value: object,
        attrs: dict[str, Any],
        shape: Sequence[int],
        dtype: object,
        *,
        schema_version: int = 1,
        name: str | None = None,
        source_location: str | None = None,
        literal_weak: bool = False,
    ) -> int: ...
    def bind_trace_frame(self, trace_level: int, trace_frame_id: int) -> None: ...
    def runtime_trace_identity(self) -> tuple[int | None, int | None]: ...
    def record_residual(self, node_id: int, residual: object) -> None: ...
    def value(self, node_id: int) -> object: ...
    def values(self, node_ids: Sequence[int]) -> list[object]: ...
    def mark_weak(self, node_id: int) -> None: ...
    def is_weak(self, node_id: int) -> bool: ...
    def node_is_active(self, node_id: int) -> bool: ...
    def weak_mask(self, node_ids: Sequence[int]) -> list[bool]: ...
    def mark_output(self, node_id: int) -> None: ...
    def freeze(
        self,
        jvp_bindings: Sequence[Callable[..., object] | None],
        vjp_bindings: Sequence[Callable[..., object] | None],
        reverse_needs: Sequence[tuple[bool, bool, bool] | None],
    ) -> None: ...
    def prune_reverse_payloads(self) -> None: ...
    def set_active_nodes(self, node_ids: Sequence[int]) -> None: ...
    def analyze_real_linearity(
        self,
        tangent_input_ids: Sequence[int],
        primitive_name: str,
    ) -> list[int]: ...
    def release_payloads(self) -> None: ...
    def get_node_name(self, node_id: int) -> str | None: ...
    def _diagnostic_snapshot(
        self,
    ) -> list[tuple[str, str | None, object]]: ...
    def stats(self) -> dict[str, object]: ...

@final
class GraphExecutionPlan: ...

@final
class GraphNode:
    @property
    def id(self) -> int: ...
    @property
    def op(self) -> str: ...
    @property
    def schema_version(self) -> int: ...
    @property
    def inputs(self) -> list[int]: ...
    @property
    def attrs(self) -> dict[str, Any]: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> object: ...
    @property
    def name(self) -> str | None: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def output_shapes(self) -> list[list[int]] | None: ...
    @property
    def output_dtypes(self) -> list[Any] | None: ...
    @property
    def source_location(self) -> str | None: ...

@final
class GraphBuilder:
    def __new__(
        cls,
        *,
        required_array_api_version: str = "2024.12",
    ) -> Self: ...
    def append_node(
        self,
        op: str,
        inputs: Sequence[int],
        attrs: dict[str, Any],
        shape: Sequence[int],
        dtype: object,
        *,
        name: str | None = None,
        num_outputs: int = 1,
        output_shapes: Sequence[Sequence[int]] | None = None,
        output_dtypes: Sequence[Any] | None = None,
        source_location: str | None = None,
        schema_version: int = 1,
    ) -> int: ...
    def append_input_node(
        self,
        shape: Sequence[int],
        dtype: object,
        *,
        name: str | None = None,
    ) -> int: ...
    def append_constant(
        self,
        data: bytes,
        shape: Sequence[int],
        dtype: object,
        *,
        kind: Literal["scalar", "array"],
    ) -> tuple[int, str]: ...
    def append_output(self, node_id: int) -> None: ...
    def finish(
        self,
    ) -> tuple[
        GraphStore,
        list[int | None],
        dict[str, Any],
        list[tuple[int, str, list[int], str | None]],
    ]: ...

@final
class GraphStore:
    @property
    def required_array_api_version(self) -> str: ...
    @property
    def node_count(self) -> int: ...
    @property
    def inputs(self) -> list[int]: ...
    @property
    def outputs(self) -> list[int]: ...
    def get_node(self, node_id: int) -> GraphNode: ...
    def node_ids(self) -> list[int]: ...
    def constant_ids(self) -> list[int]: ...
    def _to_json(self) -> str: ...
    def _constant_parts(
        self,
        node_id: int,
    ) -> tuple[str, str, list[int], bytes, str]: ...
