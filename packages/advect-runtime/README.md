# advect-runtime

`advect-runtime` is the Python-independent runtime for Advect's staged graph.
It validates and stores closed graph data, runs the fixed optimization pipeline,
builds execution schedules, and releases values after their last use. A host
supplies the numerical operations and array handles.

The crate also owns the canonical flat graph artifact. Constants become raw
bytes inside the runtime and lowercase hexadecimal only at the JSON boundary.

`advect-runtime` has no Python, PyO3, or array-library dependency. The private
`advect-native` extension supplies its Python host, while a pure Rust fixture
host proves that graph execution is independent of Python. The crate is
workspace-internal and does not yet promise a stable crates.io API.

This is an operation runtime, not a kernel compiler. Code generation, fusion,
backend lowering, automatic checkpoint placement, and numerical kernels remain
outside its scope. See the [codebase map](../../docs/development/codebase.md)
for the boundary with Python and `advect-native`.
