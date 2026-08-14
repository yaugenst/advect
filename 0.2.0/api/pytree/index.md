# Pytrees

[Pytrees](https://yaugenst.github.io/advect/0.2.0/tutorials/gradients/#select-arguments-and-preserve-structure) let transforms preserve nested tuples, lists, dictionaries, and registered application containers while differentiating the arrays inside them. Other values remain part of the structure. Custom classes may [register a node type](https://yaugenst.github.io/advect/0.2.0/api/pytree/#advect.pytree.register_pytree_node) for dynamic transforms; [staged programs](https://yaugenst.github.io/advect/0.2.0/api/staging/index.md) accept the built-in portable container forms rather than arbitrary Python class registrations.

## pytree

Public pytree utilities.

This module exposes Advect's structured tree utilities for advanced users.

### DictKey

```python
DictKey(key: Any)
```

Path entry for indexing into dict pytree nodes.

### SequenceKey

```python
SequenceKey(index: int)
```

Path entry for indexing into sequence pytree nodes.

### Static

```python
Static(value: T)
```

Wrapper for marking values as static (non-flattened) in pytrees.

### TreeDef

```python
TreeDef(
    node_type: type[Any] | None,
    aux_data: Any,
    children: tuple[TreeDef, ...],
    num_leaves: int,
)
```

A structural description of a pytree.

Attributes:

- **`node_type`** (`type[Any] | None`) – Container node type, or None for leaf nodes.
- **`aux_data`** (`Any`) – Node-specific metadata needed to reconstruct the tree.
- **`children`** (`tuple[TreeDef, ...]`) – Child TreeDefs.
- **`num_leaves`** (`int`) – Total number of leaves in this subtree.

### format_path

```python
format_path(path: TreePath) -> str
```

Format a pytree leaf path in bracket syntax.

Examples:

```pycon
>>> import advect as ad
>>> path = (ad.pytree.DictKey("params"), ad.pytree.SequenceKey(0))
>>> ad.pytree.format_path(path)
"['params'][0]"
```

### register_pytree_node

```python
register_pytree_node(
    cls: type[Any],
    *,
    flatten_fn: _FlattenFn,
    unflatten_fn: _UnflattenFn,
    include_subclasses: bool = False,
) -> None
```

Register a custom pytree node type.

Parameters:

- **`cls`** (`type[Any]`) – Class to register as a pytree node.
- **`flatten_fn`** (`_FlattenFn`) – Function flatten_fn(obj) -> (children, aux_data).
- **`unflatten_fn`** (`_UnflattenFn`) – Function unflatten_fn(aux_data, children) -> obj.
- **`include_subclasses`** (`bool`, default: `False` ) – Whether subclasses without their own registration inherit this node implementation. The nearest registered base in the method resolution order wins.

### static

```python
static(value: T) -> Static[T]
```

Wrap `value` as a static pytree node.

Static nodes have no leaves: they are preserved by `tree_map` and passed through tracing/autodiff as untraceable metadata.

Examples:

```pycon
>>> import advect as ad
>>> tree = (1, ad.pytree.static({"mode": "train"}))
>>> ad.pytree.tree_leaves(tree)
[1]
>>> mapped = ad.pytree.tree_map(lambda value: value + 1, tree)
>>> mapped[0], mapped[1].value
(2, {'mode': 'train'})
```

### tree_flatten

```python
tree_flatten(tree: Any) -> tuple[list[Any], TreeDef]
```

Flatten a pytree into a list of leaves and a TreeDef.

Examples:

```pycon
>>> import advect as ad
>>> leaves, treedef = ad.pytree.tree_flatten({"x": [1, 2]})
>>> leaves, treedef.num_leaves
([1, 2], 2)
```

### tree_flatten_with_paths

```python
tree_flatten_with_paths(
    tree: Any,
) -> tuple[list[TreePath], list[Any], TreeDef]
```

Flatten a pytree into (paths, leaves, treedef).

Paths are tuples of typed path entries describing the location of each leaf. Dict nodes use `DictKey`, and sequence-like nodes use `SequenceKey`.

Examples:

```pycon
>>> import advect as ad
>>> paths, leaves, _ = ad.pytree.tree_flatten_with_paths({"x": [1, 2]})
>>> [ad.pytree.format_path(path) for path in paths]
["['x'][0]", "['x'][1]"]
>>> leaves
[1, 2]
```

### tree_leaves

```python
tree_leaves(tree: Any) -> list[Any]
```

Return the leaves of a pytree.

Examples:

```pycon
>>> import advect as ad
>>> ad.pytree.tree_leaves({"x": [1, 2], "y": 3})
[1, 2, 3]
```

### tree_map

```python
tree_map(f: Any, tree: Any) -> Any
```

```python
tree_map(f: Any, tree: Any, /, *rest: Any) -> Any
```

```python
tree_map(f: Any, tree: Any, /, *rest: Any) -> Any
```

Apply `f` to each leaf in one or more pytrees.

When multiple trees are provided, they must have identical TreeDefs.

Examples:

```pycon
>>> import advect as ad
>>> ad.pytree.tree_map(lambda x, y: x + y, {"x": [1, 2]}, {"x": [3, 4]})
{'x': [4, 6]}
```

### tree_unflatten

```python
tree_unflatten(treedef: TreeDef, leaves: list[Any]) -> Any
```

Reconstruct a pytree from a TreeDef and a list of leaves.

Examples:

```pycon
>>> import advect as ad
>>> _, treedef = ad.pytree.tree_flatten({"x": [1, 2]})
>>> ad.pytree.tree_unflatten(treedef, [3, 4])
{'x': [3, 4]}
```
