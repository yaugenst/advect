# Arrays and Structure

Provider-preserving construction, tracer introspection, and structured
inputs/outputs.

For Array API inputs, a dynamic transform requests supported revisions newest
first and selects the newest common result for the whole call. The ordered
profiles are `2022.12`, `2023.12`, and `2024.12`; mixed providers still fail.
Providers may report a newer revision after accepting the explicit request, and
Advect retains that provider metadata rather than relabeling the namespace.

NumPy remains a separate protocol frontend. Advect supports NumPy 2.0 through
2.4. Its Array API targets are 2022.12 for NumPy 2.0, 2023.12 for NumPy 2.1-2.2,
and 2024.12 for NumPy 2.3-2.4. Live NumPy handlers define the callable surface;
there is no parallel versioned callable inventory.

::: advect.array

::: advect.asarray

::: advect.is_traced

::: advect.stop_gradient

::: advect.support_catalog

::: advect.pytree
