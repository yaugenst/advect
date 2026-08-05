# Implementation

This section records the current numerical package and its evidence contracts.

- Implementation status: `design/implementation/roadmap.md`
- Performance: `design/implementation/performance.md`
- Array provider qualification:
  `design/implementation/array-provider-qualification.md`
- Upstream compatibility:
  `design/implementation/upstream-compatibility.md`

## Current Focus

The numerical contract, singular array tracer, invocation-local primitive
residuals, shared native SSA substrate, native dynamic traversal, singular
staged programs, and fixed staged optimizer are implemented.

The lifetime-aware runtime program is also implemented. Dynamic reverse has
explicit retention and last-use release, and the staged graph, canonical
artifact, fixed cleanup, and execution planner live in the PyO3-free
`advect-runtime` crate. `advect-native` is the Python adapter. Kernel compilation,
code generation, fusion, backend lowering, automatic checkpoint placement,
and a Rust array library remain out of scope.

See [Implementation Status](roadmap.md) for the current product shape and
[Runtime and Performance](performance.md) for the measurement contract. The
[Array Provider Qualification](array-provider-qualification.md) records the
portable NumPy, Array API Strict, and single-device CuPy scientific matrix and
the generated official trace-and-execute surface. The
[Upstream Compatibility](upstream-compatibility.md) defines dependency bounds
and the executable extension-qualification contract.
