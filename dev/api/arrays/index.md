# Arrays and Structure

Provider-preserving construction, tracer introspection, and structured inputs/outputs.

For Array API inputs, a dynamic transform requests supported revisions newest first and selects the newest common result for the whole call. The ordered profiles are `2022.12`, `2023.12`, and `2024.12`; mixed providers still fail. Providers may report a newer revision after accepting the explicit request, and Advect retains that provider metadata rather than relabeling the namespace.

NumPy remains a separate protocol frontend. Advect supports NumPy 2.0 through 2.4. Its Array API targets are 2022.12 for NumPy 2.0, 2023.12 for NumPy 2.1-2.2, and 2024.12 for NumPy 2.3-2.4. Live NumPy handlers define the callable surface; there is no parallel versioned callable inventory.

## array

```python
array(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool = True,
) -> Any
```

Construct an owned array while preserving traced dependencies.

This is the explicit traced counterpart of the common `numpy.array(obj, dtype=..., copy=...)` forms. It intentionally does not mirror NumPy's complete constructor signature.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def total(value):
...     return np.sum(ad.array([value[0], value[1]]))
>>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
[1.0, 1.0]
```

## asarray

```python
asarray(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool | None = None,
) -> Any
```

Construct an array without detaching Advect tracers.

Direct tracers and rectangular nested tracer sequences remain differentiable. This is the provider-neutral explicit alternative to NumPy's standard `numpy.asarray(..., like=tracer)` dispatch. Ordinary non-traced values retain their provider when they expose the pinned Array API namespace and otherwise use NumPy.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def total(value):
...     return np.sum(ad.asarray([value[0], value[1]]))
>>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
[1.0, 1.0]
```

## is_traced

```python
is_traced(value: object) -> bool
```

Return whether `value` is an Advect tracer.

This check does not read the trace-time payload and remains safe for an escaped tracer. It tests the value itself rather than recursively searching an arbitrary object graph.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> ad.is_traced(np.array([1.0]))
False
>>> def contains_tracer(value):
...     assert ad.is_traced(value)
...     return np.sum(value**2)
>>> ad.grad(contains_tracer)(np.array([2.0])).tolist()
[4.0]
```

## stop_gradient

```python
stop_gradient(value: T) -> T
```

Return a concrete copy of traced leaves, explicitly stopping gradients.

Registered pytree structure is preserved. The operation is available only during concrete dynamic tracing; staging rejects it because an abstract value has no concrete primal to validate or serialize.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def loss(value):
...     return np.sum(value * ad.stop_gradient(value))
>>> ad.grad(loss)(np.array([2.0, 3.0])).tolist()
[2.0, 3.0]
```

## support_catalog

```python
support_catalog() -> dict[str, object]
```

Return live primitive capabilities and supported functions by extension.

Each mode marked true is an end-to-end support claim for the callable's declared frontend contract, rather than a statement that a handler exists.

Examples:

```pycon
>>> import advect as ad
>>> catalog = ad.support_catalog()
>>> catalog["schema_version"]
3
>>> sorted(catalog["extensions"])
['array_api', 'numpy', 'scipy']
```

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

| Name         | Type                  | Description                                            |
| ------------ | --------------------- | ------------------------------------------------------ |
| `node_type`  | \`type[Any]           | None\`                                                 |
| `aux_data`   | `Any`                 | Node-specific metadata needed to reconstruct the tree. |
| `children`   | `tuple[TreeDef, ...]` | Child TreeDefs.                                        |
| `num_leaves` | `int`                 | Total number of leaves in this subtree.                |

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
) -> None
```

Register a custom pytree node type.

Parameters:

| Name           | Type           | Description                                       | Default    |
| -------------- | -------------- | ------------------------------------------------- | ---------- |
| `cls`          | `type[Any]`    | Class to register as a pytree node.               | *required* |
| `flatten_fn`   | `_FlattenFn`   | Function flatten_fn(obj) -> (children, aux_data). | *required* |
| `unflatten_fn` | `_UnflattenFn` | Function unflatten_fn(aux_data, children) -> obj. | *required* |

### static

```python
static(value: T) -> Static[T]
```

Wrap `value` as a static pytree node.

Static nodes have no leaves: they are preserved by :func:`tree_map` and passed through tracing/autodiff as untraceable metadata.

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

Paths are tuples of typed path entries describing the location of each leaf. Dict nodes use :class:`DictKey`, and sequence-like nodes use :class:`SequenceKey`.

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
