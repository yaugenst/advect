# Pytrees

[Pytrees](../tutorials/gradients.md#select-arguments-and-preserve-structure) let
transforms preserve nested tuples, lists, dictionaries, and registered
application containers while differentiating the arrays inside them. Other
values remain part of the structure. Custom classes may
[register a node type](pytree.md#advect.pytree.register_pytree_node) for dynamic
transforms; [staged programs](staging.md) accept the built-in portable
container forms rather than arbitrary Python class registrations.

::: advect.pytree
