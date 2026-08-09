"""Required bindings for Advect's native dynamic tape and staged graph."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advect._native_core import (
        DynamicTape,
        GraphBuilder,
        GraphExecutionPlan,
        GraphNode as NativeNode,
        GraphStore,
        __build_profile__,
        __version__,
        build_graph_execution_plan,
        deserialize_graph_json,
        dynamic_jvp,
        dynamic_jvp_many,
        dynamic_vjp,
        dynamic_vjp_many,
        execute_graph,
    )
else:
    _native_core = import_module("advect._native_core")
    DynamicTape = _native_core.DynamicTape
    GraphBuilder = _native_core.GraphBuilder
    GraphExecutionPlan = _native_core.GraphExecutionPlan
    NativeNode = _native_core.GraphNode
    GraphStore = _native_core.GraphStore
    build_graph_execution_plan = _native_core.build_graph_execution_plan
    deserialize_graph_json = _native_core.deserialize_graph_json
    dynamic_jvp = _native_core.dynamic_jvp
    dynamic_jvp_many = _native_core.dynamic_jvp_many
    dynamic_vjp = _native_core.dynamic_vjp
    dynamic_vjp_many = _native_core.dynamic_vjp_many
    execute_graph = _native_core.execute_graph
    __build_profile__ = _native_core.__build_profile__
    __version__ = _native_core.__version__

__all__ = [
    "DynamicTape",
    "GraphBuilder",
    "GraphExecutionPlan",
    "GraphStore",
    "NativeNode",
    "build_graph_execution_plan",
    "create_graph_builder",
    "deserialize_graph_json",
    "dynamic_jvp",
    "dynamic_jvp_many",
    "dynamic_vjp",
    "dynamic_vjp_many",
    "execute_graph",
    "native_build_info",
]


def create_graph_builder(*, required_array_api_version: str) -> GraphBuilder:
    """Construct the required native append-only graph builder."""
    return GraphBuilder(required_array_api_version=required_array_api_version)


def native_build_info() -> dict[str, str]:
    """Return immutable native build provenance for diagnostics."""
    return {
        "version": __version__,
        "build_profile": __build_profile__,
    }
