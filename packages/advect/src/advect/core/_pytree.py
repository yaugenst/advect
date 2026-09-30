# ruff: noqa: ANN401
"""Pytree utilities for structured inputs and outputs.

This module implements a minimal pytree system (similar in spirit to JAX pytrees)
used to support structured inputs/outputs across Advect's tracing and autodiff APIs.

The core is stdlib-only and backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DictKey",
    "SequenceKey",
    "Static",
    "TreeDef",
    "TreePath",
    "format_path",
    "register_pytree_node",
    "static",
    "tree_flatten",
    "tree_flatten_with_paths",
    "tree_leaves",
    "tree_map",
    "tree_unflatten",
]


class _FlattenFn(Protocol):
    def __call__(self, tree: Any) -> tuple[tuple[Any, ...], Any]: ...


class _UnflattenFn(Protocol):
    def __call__(self, aux_data: Any, children: tuple[Any, ...]) -> Any: ...


_REGISTRY: dict[type[Any], tuple[_FlattenFn, _UnflattenFn]] = {}
_INHERITED_REGISTRY: set[type[Any]] = set()
_PROTOCOL_RESULT_ARITY = 2


def _tree_contains_tracer(value: Any, _seen: dict[int, Any] | None = None) -> bool:
    """Return whether leaves or registered-node metadata contain an Advect tracer."""
    if callable(getattr(value, "_advect_snapshot", None)):
        return True
    if type(value) in (type(None), bool, int, float, complex, str, bytes, bytearray):
        return False
    # Keep every visited value alive: a flatten function may build fresh child
    # containers whose addresses CPython would otherwise reuse mid-scan.
    seen = {} if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return False
    seen[identity] = value

    impl = _get_node_impl(type(value))
    if impl is not None:
        flatten_fn, _unflatten_fn = impl
        children, aux_data = flatten_fn(value)
        return any(_tree_contains_tracer(child, seen) for child in children) or (
            _tree_contains_tracer(aux_data, seen)
        )
    if isinstance(value, dict):
        return any(
            _tree_contains_tracer(key, seen) or _tree_contains_tracer(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_tree_contains_tracer(item, seen) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class DictKey:
    """Path entry for indexing into dict pytree nodes."""

    key: Any


@dataclass(frozen=True, slots=True)
class SequenceKey:
    """Path entry for indexing into sequence pytree nodes."""

    index: int


type PathEntry = DictKey | SequenceKey
type TreePath = tuple[PathEntry, ...]


def format_path(path: TreePath) -> str:
    """Format a pytree leaf path in bracket syntax.

    Examples
    --------
    >>> import advect as ad
    >>> path = (ad.pytree.DictKey("params"), ad.pytree.SequenceKey(0))
    >>> ad.pytree.format_path(path)
    "['params'][0]"
    """
    parts: list[str] = []
    for entry in path:
        if isinstance(entry, DictKey):
            parts.append(f"[{entry.key!r}]")
        else:
            parts.append(f"[{entry.index!r}]")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class TreeDef:
    """A structural description of a pytree.

    Attributes
    ----------
    node_type
        Container node type, or None for leaf nodes.
    aux_data
        Node-specific metadata needed to reconstruct the tree.
    children
        Child TreeDefs.
    num_leaves
        Total number of leaves in this subtree.
    """

    node_type: type[Any] | None
    aux_data: Any
    children: tuple[TreeDef, ...]
    num_leaves: int


@dataclass(frozen=True, slots=True)
class Static[T]:
    """Wrapper for marking values as static (non-flattened) in pytrees."""

    value: T

    def __post_init__(self) -> None:
        if _tree_contains_tracer(self.value):
            msg = (
                "Static pytree metadata cannot contain an Advect tracer. "
                "Pass the value as a dynamic pytree leaf instead."
            )
            raise TypeError(msg)


def static[T](value: T) -> Static[T]:
    """Wrap ``value`` as a static pytree node.

    Static nodes have no leaves: they are preserved by ``tree_map`` and
    passed through tracing/autodiff as untraceable metadata.

    Examples
    --------
    >>> import advect as ad
    >>> tree = (1, ad.pytree.static({"mode": "train"}))
    >>> ad.pytree.tree_leaves(tree)
    [1]
    >>> mapped = ad.pytree.tree_map(lambda value: value + 1, tree)
    >>> mapped[0], mapped[1].value
    (2, {'mode': 'train'})
    """
    return Static(value)


def register_pytree_node(
    cls: type[Any],
    *,
    flatten_fn: _FlattenFn,
    unflatten_fn: _UnflattenFn,
    include_subclasses: bool = False,
) -> None:
    """Register a custom pytree node type.

    Parameters
    ----------
    cls
        Class to register as a pytree node.
    flatten_fn
        Function ``flatten_fn(obj) -> (children, aux_data)``.
    unflatten_fn
        Function ``unflatten_fn(aux_data, children) -> obj``.
    include_subclasses
        Whether subclasses without their own registration inherit this node
        implementation. The nearest registered base in the method resolution
        order wins.
    """
    _REGISTRY[cls] = (flatten_fn, unflatten_fn)
    if include_subclasses:
        _INHERITED_REGISTRY.add(cls)
    else:
        _INHERITED_REGISTRY.discard(cls)


def _dict_flatten(tree: dict[Any, Any]) -> tuple[tuple[Any, ...], Any]:
    keys = tuple(tree.keys())
    children = tuple(tree[k] for k in keys)
    return children, keys


def _dict_unflatten(aux_data: Any, children: tuple[Any, ...]) -> dict[Any, Any]:
    keys = aux_data
    if not isinstance(keys, tuple):
        msg = (
            f"Invalid treedef aux_data for dict: expected tuple of keys, got {type(keys).__name__}"
        )
        raise TypeError(msg)
    return dict(zip(keys, children, strict=True))


def _list_flatten(tree: list[Any]) -> tuple[tuple[Any, ...], Any]:
    return tuple(tree), len(tree)


def _list_unflatten(aux_data: Any, children: tuple[Any, ...]) -> list[Any]:
    _ = aux_data
    return list(children)


def _tuple_flatten(tree: tuple[Any, ...]) -> tuple[tuple[Any, ...], Any]:
    return tree, len(tree)


def _tuple_unflatten(aux_data: Any, children: tuple[Any, ...]) -> tuple[Any, ...]:
    _ = aux_data
    return tuple(children)


def _static_flatten(tree: Static[Any]) -> tuple[tuple[Any, ...], Any]:
    return (), tree.value


def _static_unflatten(aux_data: Any, children: tuple[Any, ...]) -> Static[Any]:
    if children:
        msg = "Static pytree node must have no children"
        raise ValueError(msg)
    return Static(aux_data)


register_pytree_node(dict, flatten_fn=_dict_flatten, unflatten_fn=_dict_unflatten)
register_pytree_node(list, flatten_fn=_list_flatten, unflatten_fn=_list_unflatten)
register_pytree_node(tuple, flatten_fn=_tuple_flatten, unflatten_fn=_tuple_unflatten)
register_pytree_node(Static, flatten_fn=_static_flatten, unflatten_fn=_static_unflatten)


def _get_node_impl(node_type: type[Any]) -> tuple[_FlattenFn, _UnflattenFn] | None:
    registered = _REGISTRY.get(node_type)
    if registered is not None:
        return registered

    protocol_flatten = getattr(node_type, "__advect_tree_flatten__", None)
    protocol_unflatten = getattr(node_type, "__advect_tree_unflatten__", None)
    if (protocol_flatten is None) != (protocol_unflatten is None):
        msg = (
            f"Pytree protocol on {node_type.__name__} requires both "
            "__advect_tree_flatten__ and __advect_tree_unflatten__"
        )
        raise TypeError(msg)
    if protocol_flatten is not None and protocol_unflatten is not None:
        if not callable(protocol_flatten) or not callable(protocol_unflatten):
            msg = f"Pytree protocol hooks on {node_type.__name__} must be callable"
            raise TypeError(msg)

        def flatten_protocol(tree: Any) -> tuple[tuple[Any, ...], Any]:
            result = tree.__advect_tree_flatten__()
            if not isinstance(result, tuple) or len(result) != _PROTOCOL_RESULT_ARITY:
                msg = (
                    f"{node_type.__name__}.__advect_tree_flatten__() "
                    "must return (children, aux_data)"
                )
                raise TypeError(msg)
            children, aux_data = result
            if not isinstance(children, tuple):
                msg = f"{node_type.__name__}.__advect_tree_flatten__() children must be a tuple"
                raise TypeError(msg)
            return children, aux_data

        def unflatten_protocol(aux_data: Any, children: tuple[Any, ...]) -> Any:
            return node_type.__advect_tree_unflatten__(aux_data, children)

        return flatten_protocol, unflatten_protocol

    for base in node_type.__mro__[1:]:
        if base in _INHERITED_REGISTRY:
            return _REGISTRY[base]

    if issubclass(node_type, tuple) and isinstance(getattr(node_type, "_fields", None), tuple):

        def flatten_namedtuple(tree: Any) -> tuple[tuple[Any, ...], Any]:
            return tuple(tree), None

        def unflatten_namedtuple(aux_data: Any, children: tuple[Any, ...]) -> Any:
            _ = aux_data
            return node_type(*children)

        return flatten_namedtuple, unflatten_namedtuple
    return None


_LEAF_TREEDEF = TreeDef(node_type=None, aux_data=None, children=(), num_leaves=1)
_MISSING_LEAF = object()


def _flatten(
    tree: Any,
    leaves: list[Any],
    paths: list[TreePath] | None,
    path: TreePath,
) -> TreeDef:
    """Append ``tree``'s leaves, and their paths when requested, in pytree order."""
    node_type = type(tree)
    impl = _get_node_impl(node_type)
    if impl is None:
        leaves.append(tree)
        if paths is not None:
            paths.append(path)
        return _LEAF_TREEDEF

    children, aux_data = impl[0](tree)
    if paths is None:
        child_defs = [_flatten(child, leaves, None, path) for child in children]
    elif node_type is dict:
        if not isinstance(aux_data, tuple) or len(aux_data) != len(children):
            msg = "Invalid dict pytree aux_data: expected tuple of keys matching children"
            raise TypeError(msg)
        child_defs = [
            _flatten(child, leaves, paths, (*path, DictKey(key)))
            for key, child in zip(aux_data, children, strict=True)
        ]
    else:
        child_defs = [
            _flatten(child, leaves, paths, (*path, SequenceKey(index)))
            for index, child in enumerate(children)
        ]
    num_leaves = 0
    for child_def in child_defs:
        num_leaves += child_def.num_leaves
    return TreeDef(node_type, aux_data, tuple(child_defs), num_leaves)


def tree_flatten(tree: Any) -> tuple[list[Any], TreeDef]:
    """Flatten a pytree into a list of leaves and a TreeDef.

    Examples
    --------
    >>> import advect as ad
    >>> leaves, treedef = ad.pytree.tree_flatten({"x": [1, 2]})
    >>> leaves, treedef.num_leaves
    ([1, 2], 2)
    """
    leaves: list[Any] = []
    treedef = _flatten(tree, leaves, None, ())
    return leaves, treedef


def tree_flatten_with_paths(tree: Any) -> tuple[list[TreePath], list[Any], TreeDef]:
    """Flatten a pytree into (paths, leaves, treedef).

    Paths are tuples of typed path entries describing the location of each leaf.
    Dict nodes use ``DictKey``, and sequence-like nodes use ``SequenceKey``.

    Examples
    --------
    >>> import advect as ad
    >>> paths, leaves, _ = ad.pytree.tree_flatten_with_paths({"x": [1, 2]})
    >>> [ad.pytree.format_path(path) for path in paths]
    ["['x'][0]", "['x'][1]"]
    >>> leaves
    [1, 2]
    """
    paths: list[TreePath] = []
    leaves: list[Any] = []
    treedef = _flatten(tree, leaves, paths, ())
    return paths, leaves, treedef


def _unflatten(treedef: TreeDef, leaves: Iterator[Any]) -> Any:
    node_type = treedef.node_type
    if node_type is None:
        leaf = next(leaves, _MISSING_LEAF)
        if leaf is _MISSING_LEAF:
            # Only a hand-built TreeDef can undercount its children's leaves.
            msg = "Unflatten needs more leaves than its treedef declares"
            raise ValueError(msg)
        return leaf
    impl = _get_node_impl(node_type)
    if impl is None:
        msg = f"Unregistered pytree node type: {node_type.__name__}"
        raise TypeError(msg)
    children = [_unflatten(child_def, leaves) for child_def in treedef.children]
    return impl[1](treedef.aux_data, tuple(children))


def tree_unflatten(treedef: TreeDef, leaves: list[Any]) -> Any:
    """Reconstruct a pytree from a TreeDef and a list of leaves.

    Examples
    --------
    >>> import advect as ad
    >>> _, treedef = ad.pytree.tree_flatten({"x": [1, 2]})
    >>> ad.pytree.tree_unflatten(treedef, [3, 4])
    {'x': [3, 4]}
    """
    if treedef.num_leaves != len(leaves):
        msg = f"treedef expects {treedef.num_leaves} leaves, got {len(leaves)}"
        raise ValueError(msg)
    if treedef.node_type is None:
        return leaves[0]

    remaining = iter(leaves)
    result = _unflatten(treedef, remaining)
    unused = sum(1 for _leaf in remaining)
    if unused:
        consumed = len(leaves) - unused
        msg = f"Unflatten did not consume all leaves: consumed {consumed}, total {len(leaves)}"
        raise ValueError(msg)
    return result


def tree_leaves(tree: Any) -> list[Any]:
    """Return the leaves of a pytree.

    Examples
    --------
    >>> import advect as ad
    >>> ad.pytree.tree_leaves({"x": [1, 2], "y": 3})
    [1, 2, 3]
    """
    leaves, _treedef = tree_flatten(tree)
    return leaves


@overload
def tree_map(f: Any, tree: Any, /) -> Any: ...


@overload
def tree_map(f: Any, tree: Any, /, *rest: Any) -> Any: ...


def tree_map(f: Any, tree: Any, /, *rest: Any) -> Any:
    """Apply ``f`` to each leaf in one or more pytrees.

    When multiple trees are provided, they must have identical TreeDefs.

    Examples
    --------
    >>> import advect as ad
    >>> ad.pytree.tree_map(lambda x, y: x + y, {"x": [1, 2]}, {"x": [3, 4]})
    {'x': [4, 6]}
    """
    leaves0, treedef0 = tree_flatten(tree)
    if not rest:
        return tree_unflatten(treedef0, [f(x) for x in leaves0])

    all_leaves = [leaves0]
    for other in rest:
        leaves_i, treedef_i = tree_flatten(other)
        if treedef_i != treedef0:
            msg = "tree_map requires all input trees to have the same structure"
            raise ValueError(msg)
        all_leaves.append(leaves_i)

    mapped = [f(*xs) for xs in zip(*all_leaves, strict=True)]
    return tree_unflatten(treedef0, mapped)
