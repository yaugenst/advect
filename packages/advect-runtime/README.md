# advect-runtime

`advect-runtime` is Advect's Python-independent staged SSA graph and lifetime
runtime. It owns closed graph data, validation, optimization, serialization,
execution schedules, and host-independent value lifetimes. Numerical kernels
are supplied by a host.

The crate contains:

- `RawArena`, `GraphBuilder`, and immutable `GraphStore`;
- closed attributes, dtypes, topology, and portable numeric constants;
- canonical graph artifact version 2.0 with transactional validation;
- the fixed `DCE -> simplify -> CSE` cleanup;
- the host-independent `ExecutionPlan`, its retained `Arc<GraphStore>`,
  remaining-use counts, conservative alias-root sets, last-use release, and
  donation selection;
- the `Host` contract for linking operations, materializing constants,
  retaining handles for repeated flat outputs, evaluating opaque values, and
  validating every source kind and multi-output leaf before declaring ownership
  or aliases.

Live constants are raw bytes. Their canonical JSON form uses lowercase
hexadecimal only at the artifact boundary.

`advect-runtime` has no PyO3 or Python dependency and contains no numerical array
library. `advect-native` supplies the Python host adapter; a pure Rust fixture
host proves independent graph execution. The crate is workspace-internal and
does not yet promise a stable crates.io API.

This runtime interprets host operations. It is not a kernel compiler and has no
code generation, fusion, backend lowering, automatic checkpoint planner, or
Rust NumPy implementation.
