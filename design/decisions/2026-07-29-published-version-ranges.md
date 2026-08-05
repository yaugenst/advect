# ADR: Published Version Ranges and Qualified Upstream Profiles

**Date:** 2026-07-29
**Status:** Superseded by [Runtime-Derived Extension Catalogs](2026-07-31-runtime-derived-extension-catalogs.md)
**Implementation status:** Not implemented
**Amends:** [Versioned Upstream Callable Contracts](2026-07-29-versioned-upstream-contracts.md),
section “Upstream versions are bounded”

## Context

Advect should fit into an existing scientific Python environment without forcing
users onto one patch release or one recent monthly release of every optional
dependency. Exact pins are useful for reproducing a qualification run, but
they are a poor package interface: they create resolver conflicts, block
upstream bug fixes, and make an Advect installation unnecessarily determine the
rest of a user's environment.

At the same time, a broad dependency specifier is not evidence that every
upstream callable has one stable contract across that range. NumPy adds
functions and changes signatures between minor releases. SciPy call contracts
can also evolve. Xarray and `array-api-compat` present a different problem:
Advect uses a small integration boundary from each package rather than claiming
their full public namespaces.

We therefore separate three concepts:

- the **published range** controls which environments can install Advect;
- an **upstream profile** records the callable or integration contract that
  applies to part of that range;
- **qualification evidence** records the exact versions on which that contract
  was executed.

The published range is a user-facing compatibility promise. Exact evidence
versions remain reproducibility data, not dependency constraints.

## Decision

### Publish broad, meaningful ranges

The initial target ranges are:

| Package | Published target | Contract boundary |
|---|---|---|
| NumPy | `>=2,<3` | First-class NumPy frontend, with a callable profile for each distinct NumPy minor contract |
| SciPy | `>=1.13,<2` | The explicit `advect.scipy` allowlist, profiled where its admitted signatures differ |
| xarray | `>=2024.6` | Dynamic `DataArray` and `Dataset` pytree integration only |
| `array-api-compat` | `>=1.11.2,<2` | Base-dependency provider resolution for the Array API 2024.12 contract |

NumPy 2.0 is the semantic floor. It gives Advect one promotion model and the
modern array API conventions without introducing a parallel NumPy 1.x
compatibility layer. Supporting NumPy 1.26 would require separate weak-scalar,
promotion, constructor, and namespace behavior; that is a different product
commitment and is outside this decision.

SciPy 1.13 is the floor because it is the first SciPy release intended to work
with NumPy 2. Advect's SciPy namespace remains deliberately small: the range
applies to the named callables and solver adapters, not to arbitrary
`scipy.*` tracing.

Xarray uses calendar versions rather than a semantic major series. Advect relies
on a narrow, public container contract, so a speculative annual upper bound
would create more user friction than safety. We will test the lower bound and
the latest release and introduce an exclusion only for an observed
incompatibility.

`array-api-compat` 1.11 introduced the Array API 2024.12 profile; 1.11.2 is the
floor because it includes relevant `result_type` and `clip` corrections. Advect
contracts against the Array API revision, not against the wrapper's entire
namespace. Later 1.x releases may target newer revisions while still serving a
2024.12 request; qualification verifies that behavior.

`array-api-strict` remains an exactly pinned development reference provider.
It is not a runtime dependency and its pin does not constrain users. CuPy also
remains outside Advect's dependency metadata because users must select a build
matching their CUDA environment; exact CuPy versions belong in provider
evidence.

### Make profiles select the behavioral contract

The support manifest will distinguish a package range from an API profile. A
NumPy installation, for example, may select a 2.0, 2.1, 2.2, or 2.3 callable
profile while all four satisfy the same `numpy>=2,<3` package requirement.
Functions that did not exist in an older release are absent from that release's
profile rather than presented as Advect failures.

Profiles remain parameter-level contracts. Each records the upstream
signature, parameter roles, complete call forms, and qualified dynamic,
staged, and serialized lifetimes. When two upstream minors have the same
contract they may share the same underlying declaration; a signature or
semantic difference creates a distinct profile.

The integration records for xarray and `array-api-compat` are capability
contracts instead:

- xarray qualification covers pytree flattening, reconstruction, coordinate
  preservation, named dimensions, and dynamic differentiation of supported
  array payloads;
- `array-api-compat` qualification covers provider discovery and the complete
  Array API 2024.12 program on each admitted provider.

Neither record implies callable coverage of the package's full namespace.

### Keep version differences at frontend boundaries

Advect will not reference late-added NumPy functions unconditionally during
import. The NumPy frontend registers an upstream callable only when the
installed version exposes it. Internal derivative and staging rules continue
to use stable canonical operations, so additions such as `unstack`, `matvec`,
and `vecmat` do not determine whether Advect itself can load.

Feature detection decides whether an upstream name exists. Version selection
decides which frozen contract applies. We will not scatter version checks
through derivative rules or maintain parallel implementations when one stable
lowering is sufficient.

The package range promises that Advect imports and that every callable marked
complete for the selected profile satisfies its contract. It does not claim
that every NumPy or SciPy function is differentiable. New upstream callables
remain outside the profile until admitted, and a changed callable fails closed
in the manifest until its new parameter contract is qualified.

### Qualify ranges, not just one environment

CI will exercise the range at representative boundaries:

| Integration | Required environments |
|---|---|
| NumPy | Latest patch of every supported 2.x minor, plus the newest available 2.x release |
| SciPy | Oldest compatible NumPy/SciPy pair, current locked pair, and newest compatible pair |
| xarray | Declared lower bound with the oldest supported NumPy, and latest xarray with latest NumPy |
| `array-api-compat` | Declared lower bound and latest 1.x against the official Array API suite and each qualified provider |

Each lane runs the applicable callable conformance, dynamic/staged/serialized
round trips, and integration journeys. The ordinary development lock remains
one reproducible current environment; it does not replace the range matrix.

Pre-release CI will test upcoming NumPy and SciPy minors before they become
stable. A new minor triggers a generated signature and inventory diff. If the
existing contract remains valid, the new version joins the appropriate
profile. If it changes, the same change adds or revises the profile and its
evidence. Patch releases normally require no contract update; an observed
broken patch may be excluded explicitly.

## Critical user journeys

### Existing NumPy 2 environment

A user with NumPy 2.0, 2.1, or 2.2 installs Advect without upgrading NumPy merely
to satisfy Advect's metadata. Advect imports successfully, selects the matching
NumPy profile, and exposes the callable contracts available in that upstream
release.

### Current scientific environment

A user installs `advect[scipy,xarray]` into a current environment. The resolver
selects a mutually compatible NumPy/SciPy pair inside the broad ranges, while
xarray is not forced to one monthly release. `advect.support_catalog()` reports
the functions available from the installed numerical extensions; provider and
container qualification remain separate evidence.

### Portable CPU or GPU program

A user installs `advect` — `array-api-compat` is a base dependency with a
supported 1.x range — and supplies NumPy or CuPy arrays. Advect requests the
2024.12 namespace explicitly. Provider qualification, rather than the wrapper
package version alone, determines whether dynamic and staged execution are
claimed.

### New upstream minor

A new NumPy or SciPy minor does not require an emergency dependency-only Advect
release. Pre-release qualification identifies the contract diff, and Advect
either reuses an existing profile or adds the new one. Exact versions remain
visible in evidence without becoming installation pins.

## Implementation

1. Extend the support manifest with published version specifiers and
   version-selectable upstream profiles; keep exact tested versions in
   qualification evidence.
2. Remove import-time assumptions about NumPy functions added after 2.0 and
   add frozen 2.0, 2.1, and 2.2 callable profiles alongside the current 2.3
   profile.
3. Qualify the seven SciPy callables and solver adapters from SciPy 1.13
   through the current release, splitting profiles only where the admitted
   contract differs.
4. Replace the xarray monthly pin with its integration range and test the
   lower/latest matrix.
5. Qualify `array-api-compat` 1.11.2 and latest 1.x against Array API 2024.12,
   NumPy, and CuPy, then publish the range.
6. Add lower-bound, per-minor, latest, and pre-release dependency lanes. Widen
   `pyproject.toml` only when the corresponding lanes pass.
7. Update the compatibility report and user documentation to show published
   ranges, selected profiles, and exact evidence versions as separate fields.

## Consequences

Users gain a package that composes with ordinary scientific environments
instead of replacing their dependency choices. Advect retains an exact,
inspectable behavioral promise because broad installation ranges select
versioned contracts rather than weakening them.

The cost is a wider qualification matrix and a small amount of
version-specific frontend data. That work is preferable to compatibility
branches in the autodiff engine or restrictive dependency metadata. Exact
pins remain where they provide value—in locks, reference providers, and
evidence—and disappear from the published runtime interface.
