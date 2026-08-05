"""Public pytree utilities.

This module exposes Advect's structured tree utilities for advanced users.
"""

from __future__ import annotations

from advect.core._pytree import (
    DictKey,
    SequenceKey,
    Static,
    TreeDef,
    TreePath,
    format_path,
    register_pytree_node,
    static,
    tree_flatten,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_unflatten,
)

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
