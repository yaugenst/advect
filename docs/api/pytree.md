# Pytrees

Pytrees let transforms preserve nested tuples, lists, dictionaries, and
registered application containers while differentiating their array leaves.
Static values remain structural metadata. Custom classes may register a node
type or provide Advect's inherited flatten/unflatten hooks.

::: advect.pytree
