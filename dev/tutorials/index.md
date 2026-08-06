# Tutorials

Advect has two execution modes. Use `grad` directly when ordinary Python control flow and a fresh trace per call are what you want. Use `stage` when one shape/dtype signature will run repeatedly or when you need a serializable program.

The tutorials are built from runnable snippets. Press `[ run ]` under a snippet to execute it in your browser; each page is one Python session, and running a snippet first runs the ones above it.

- [Gradients and pytrees](https://yaugenst.github.io/advect/dev/tutorials/gradients/index.md) — `grad`, scalar inputs, traced constructors, multiple arguments, `jacobian`
- [Control flow and mutation](https://yaugenst.github.io/advect/dev/tutorials/control-flow/index.md) — loops, in-place updates, converged solves, SciPy callbacks
- [Staging and serialization](https://yaugenst.github.io/advect/dev/tutorials/staging/index.md) — `stage`, derivative programs, durable artifacts
- [Providers and interop](https://yaugenst.github.io/advect/dev/tutorials/interop/index.md) — provider-neutral functions, Array API providers, xarray labels
- [Custom primitives](https://yaugenst.github.io/advect/dev/tutorials/primitives/index.md) — `@ad.primitive` and the authoring checks
- [Debugging](https://yaugenst.github.io/advect/dev/tutorials/debugging/index.md) — scoped inspection, numerical failures, staged context, and suspicious-gradient checks
