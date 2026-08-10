# Pytrees

Pytrees let transforms preserve nested tuples, lists, dictionaries, and
registered application containers while differentiating their array leaves.
Static values remain structural metadata. Custom classes may register a node
type for dynamic transforms; staged programs accept the built-in portable
container forms rather than arbitrary Python class registrations.

::: advect.pytree
