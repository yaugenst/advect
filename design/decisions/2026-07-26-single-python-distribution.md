# ADR: One Python Distribution

**Date:** 2026-07-26
**Status:** Accepted (amended 2026-08-01: `array-api-compat` is a base
dependency and its private fallback is configured when `advect` is imported.
SciPy, xarray, and host-autodiff bridges remain opt-in extras.)

## Context

Advect's built-in Python modules share one public API, version, release cadence,
and compatibility contract. Publishing the core, autodiff, NumPy frontend, and
optional integrations as separate distributions would add coordinated pins,
entry-point discovery, and partial-install states without creating useful
product boundaries.

Third-party dependency policy is a different concern from Advect code
ownership. NumPy and `array-api-compat` implement the baseline array boundary;
SciPy, xarray, and host autodiff frameworks remain opt-in without requiring
separately versioned Advect artifacts.

## Decision

Advect publishes one Python distribution named `advect`.

The distribution contains the built-in `advect.core`, `advect.autodiff`,
`advect.numpy`, `advect.scipy`, `advect.xarray`, and `advect.interop` module
boundaries plus a private compatibility-provider bridge. Those modules retain
their current ownership and dependency direction; in particular, the internal
core layer remains stdlib-only. This is a packaging consolidation, not a merger
of runtime responsibilities or lifetimes.

NumPy and `array-api-compat` are base dependencies. Extras add optional
third-party dependencies:

- `advect[scipy]` adds SciPy;
- `advect[xarray]` adds xarray;
- `advect[scipy,xarray]` combines those two extras;
- `advect[torch]`, `advect[jax]`, and `advect[autograd]` add one host autodiff
  framework each. There is no aggregate integration extra.

Every Advect wheel contains the built-in integration modules regardless of which
extras were selected. An extra controls whether its third-party dependency is
installed, not whether Advect ships a second distribution.

The Rust `advect-runtime` and `advect-native` crates remain separate internal
crates. `advect-runtime` stays free of Python, while `advect-native` remains the
PyO3 adapter bundled into the Python wheel. Their ownership and portability
contracts do not follow the Python publication boundary.

Arrays implementing `__array_namespace__` enter through the generic frontend
directly. Arrays recognized by `array-api-compat` use one deterministic private
fallback configured by Advect itself. Advect does not discover entry points or
let an installed distribution mutate input dispatch implicitly; a dedicated
provider frontend requires a concrete supported contract before adding another
registration boundary.

## Consequences

Users install, upgrade, and report one Advect version. Base import initializes
NumPy and `array-api-compat`; optional SciPy, xarray, JAX, PyTorch, and HIPS
Autograd imports still fail clearly when their third-party dependency is absent.

Release engineering produces one Python sdist and wheel family instead of a
lockstep set of built-in distributions. Internal Python modules and Rust crates
can continue to be tested and owned independently. This decision does not
change dynamic-tape lifetimes, staged artifact semantics, derivative rules,
or provider qualification.
