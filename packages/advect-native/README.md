# advect-native

`advect-native` is Advect's private PyO3 extension. Maturin packages it as
`advect._native_core` inside the `advect` wheel; it is not installed separately.

The extension owns two Python-facing runtime jobs:

- the invocation-local dynamic tape used by JVP and VJP traversal;
- translation between Python values and the host-independent staged runtime.

It also maps exceptions, invokes Python callbacks, and links provider
operations through `PythonHost`. `advect-runtime` remains the authority for the
staged graph, validation, optimization, serialization, scheduling, and value
lifetimes. The adapter keeps only thin handles to those runtime-owned objects.

Dynamic traces are not converted into staged graphs or sent through the staged
optimizer before a backward pass. The [codebase map](../../docs/development/codebase.md)
explains the complete Python/runtime split.
