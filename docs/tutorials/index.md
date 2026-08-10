# Tutorials

Advect has two execution modes. Use `grad` directly when ordinary Python
control flow and a fresh trace per call are what you want. Use `stage` when
one shape/dtype signature will run repeatedly or when you need a serializable
program.

The tutorials are built from runnable snippets. Press `[ run ]` under a
snippet to execute it in your browser; each page is one Python session, and
running a snippet first runs the ones above it.

- [Gradients and pytrees](gradients.md) — `grad`, scalar inputs, traced
  constructors, multiple arguments
- [Dynamic control flow and mutation](control-flow.md) — data-dependent
  branches and loops, local array updates, converged solves
- [Advanced differentiation](advanced-differentiation.md) — Jacobians, nested
  derivatives, Hessians, Hessian-vector products, checkpointing
- [Staging and serialization](staging.md) — `stage`, derivative programs,
  durable artifacts
- [Providers and interop](interop.md) — provider-neutral functions, Array API
  providers, xarray labels
- [Custom primitives](primitives.md) — `@ad.primitive` and the authoring
  checks
- [Debugging](debugging.md) — scoped inspection, numerical failures, staged
  context, and suspicious-gradient checks
