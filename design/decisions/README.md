# Architecture Decisions

These decisions preserve the rationale behind Advect's current numerical
architecture. Superseded decisions remain only when their transition explains
an important current boundary.

## Decisions

- [ADR: Extensible Autodiff Core](2026-07-21-extensible-autodiff-core.md) - Concrete ephemeral tapes, abstract staging, tracer-functionalized mutation, real-linear autodiff, and Array API/NumPy frontends
- [ADR: Runtime Extension Boundaries](2026-07-24-runtime-extension-boundaries.md) - Storage and workflow boundaries, dynamic rematerialization, graph-transform policy, named views, buffer donation, and CuPy compatibility
- [ADR: Staged Differentiation](2026-07-24-staged-differentiation.md) - Lifetime-preserving `grad` and `value_and_grad` over optimized, serializable staged programs
- [ADR: Lifetime-Aware Portable Rust Runtime](2026-07-24-lifetime-aware-portable-rust-runtime.md) - Implemented reverse lifetime management and a PyO3-free staged host runtime, without a kernel compiler
- [ADR: Implicit Differentiation and Scientific Frontends](2026-07-26-scientific-extension-contracts.md) - One dynamic implicit-root transform, a bounded SciPy primitive frontend, and dynamic labeled xarray pytrees
- [ADR: One Python Distribution](2026-07-26-single-python-distribution.md) - One `advect` release artifact with dependency extras, retained internal modules and Rust crates, and explicitly imported provider adapters
- [ADR: Primitive Conformance Testing](2026-07-27-primitive-conformance-testing.md) - Three test tiers from one declaration, a named law battery, registry-joined coverage, restored Hypothesis strategies and telemetry, and a test-internal harness
- [ADR: Runtime-Derived Extension Catalogs](2026-07-31-runtime-derived-extension-catalogs.md) - Per-extension support modes discovered from live bindings and qualified end to end, replacing parameter-level manifests and profiles
- [ADR: Definition-Driven Operation Authoring](2026-07-31-definition-driven-operation-authoring.md) - Complete built-in operation records, generated conventional bindings, and no lazy registry repair
- [ADR: Host Autodiff VJP Bridges](2026-08-01-host-autodiff-vjp-bridges.md) - Explicit first-order PyTorch, JAX, and HIPS Autograd boundaries without provider registration
- [ADR: Bounded SciPy Filter Coverage](2026-08-01-scipy-filter-coverage.md) - Explicit high-value special and ndimage functions built from shared stencil and selection rules
- [ADR: Concrete NumPy and provider-neutral Array API frontends](2026-08-01-concrete-numpy-frontend.md) - Concrete NumPy protocol ownership, provider-neutral Array API execution, and canonical-operation convergence
