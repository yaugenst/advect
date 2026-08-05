# ADR: Bounded SciPy Filter Coverage

**Date:** 2026-08-01
**Status:** Accepted
**Implementation status:** Implemented

## Context

The first `advect.scipy` contract proved that optional scientific functions can
reuse Advect's ordinary primitive, staging, serialization, and provider
boundaries. It deliberately admitted only seven special functions while those
boundaries settled. Common scientific and inverse-design programs also depend
on local image filters and mathematical morphology. Calling SciPy directly at
those points breaks tracing even though most derivatives are linear stencils or
simple selected-value subgradients.

Mirroring all of SciPy would recreate a compatibility layer with unclear
semantics. Adding a separate image-processing derivative engine would duplicate
the core. The useful boundary is an explicit group of frequent functions whose
forward contract can remain SciPy-owned and whose derivatives share a small
amount of traceable numerical machinery.

## Decision

`advect.scipy` gains an explicit SciPy 1.18 surface rather than a general
interceptor. The special-function set expands with complementary error,
inverse-error, log-probability, inverse-normal, and softmax forms. A new
`advect.scipy.ndimage` module supplies 27 named filters and morphology
operations listed in the target API and runtime-derived support catalog.

Atomic calls delegate concrete evaluation to SciPy and use the existing custom
primitive contract for abstract evaluation and durable operation identity.
Normalization functions retain atomic dtype resolution, while public image
operations that SciPy defines as compositions lower through the same filter
primitives. There is no new core operation family, graph format, or runtime
evaluator.

One gather-based stencil engine defines boundary sampling for `reflect`,
`mirror`, `nearest`, `wrap`, and `constant`, including the three `grid-*`
aliases. Gaussian, uniform, convolution, and correlation rules reuse it.
Their explicit adjoints call the opposite native SciPy stencil and fold its
padding exactly; we do not assume a filter is self-adjoint at boundaries.
Convolution and correlation weights and constant boundary values remain live
operands.

One neighborhood-selection engine defines extrema, rank, and greyscale
morphology derivatives. At an equal-valued selection it shares the tangent
equally across winning window slots. Slots, rather than unique input indices,
are the contract: reflected or clamped duplicates therefore retain their
natural multiplicity. Non-flat greyscale structures are live operands. NaN
selection remains SciPy's undefined forward territory and gains no stronger
gradient promise.

Reverse execution uses explicit adjoints rather than transposing expanded
gather graphs. Linear filters call SciPy for the opposite stencil and fold the
boundary padding exactly. Selection filters recover unique winner routes from
the SciPy output. Low-cardinality full-footprint plateaus use separable counts,
and common non-flat structures reconstruct winners from padded views. Irregular
or high-cardinality ties retain an exact candidate-routing fallback. These
paths avoid footprint-dependent work in ordinary flat morphology without
weakening the equal-share contract.

Binary morphology remains outside the admitted surface. Its discrete,
data-dependent masks and repeated-until-stable iterations provide no useful
gradient contract, so staging it would add a separate nondifferentiable
operation family without advancing Advect's purpose.

All admitted functions are NumPy-backed and implement their public SciPy 1.18
signature, axes/origin/radius and footprint configuration, output-array
functionalization, and output-dtype behavior. Configuration is concrete while
tracing; array operands such as convolution weights, greyscale structures, and
constant boundary values remain graph operands. Artifact loading still requires
importing `advect.scipy` so the stable primitive schemas are linked.

## Consequences

Frequently used filtering and morphology code can now differentiate, stage,
and serialize without replacing SciPy calls with project-specific helpers.
The implementation cost is concentrated in two reusable numerical engines,
while SciPy remains the forward-semantics authority.

The equal-share rule makes morphology plateaus deterministic but is one chosen
subgradient at nonsmooth points. Providers other than NumPy, unlisted
`ndimage` functions, labeled image containers, and general SciPy interception
remain outside this decision.

Exact fallback routing for irregular footprints and high-cardinality ties
scales with the number of active footprint entries. The reproducible runtime
matrix is `uv run python -m scripts.bench_scipy_runtime --check`; its default
gate includes large unique-value footprints, tied plateaus, non-flat
structures, and constant boundaries. Ratios remain workload-specific and
should also be checked at the shapes and neighborhoods used by an application.

This decision extends the SciPy section of
[Implicit Differentiation and Scientific Frontends](2026-07-26-scientific-extension-contracts.md);
the solver and xarray decisions there are unchanged.
