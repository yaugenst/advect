# Codebase Map

Advect has one inward dependency direction: public and optional frontends adapt user calls into provider-neutral core semantics; the core never reaches back out to a frontend. Dynamic autodiff and staging share canonical operation semantics, but keep their lifetime machinery separate.

```text
user code
    |
    +-- NumPy / Array API / SciPy frontends
    +-- xarray structure / host-autodiff adapters
                    |
                    v
        canonical operation emission
                    |
          +---------+----------+
          |                    |
          v                    v
    dynamic autodiff     abstract staging
          |                    |
          v                    v
  native dynamic tape   advect-runtime graph
                              ^
                              |
                       advect-native host
```

## Source ownership

| Path                                            | Owns                                                                                                                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/advect/src/advect/`                   | Public package assembly, explicit array construction, pytrees, support reporting, and author-facing testing helpers                                                                                                               |
| `advect/core/`                                  | Standard-library-only contracts: registry, tracing context, abstract values, pytrees, diagnostics, staging, the versioned outer `StagedProgram` envelope and cross-record validation, and the provider-neutral Array API frontend |
| `advect/core/_array_api/`                       | Array API frontend behavior, revision profiles, signatures, results, provider negotiation, executable evidence, and support declarations                                                                                          |
| `advect/autodiff/`                              | Transform APIs across dynamic calls and staged programs, concrete trace lifetimes, and canonical derivative rules                                                                                                                 |
| `advect/autodiff/rules/array_family/{jvp,vjp}/` | JVP-first array-family formulas and the smaller set of explicit real-adjoint rules                                                                                                                                                |
| `advect/numpy/`                                 | NumPy protocols, traced-array behavior, exact NumPy signatures, mutation functionalization, evaluation, and public NumPy support declarations                                                                                     |
| `advect/numpy/_array_function/`                 | NumPy array-function implementations grouped by responsibility, including traceable composites                                                                                                                                    |
| `advect/scipy/`                                 | The bounded optional SciPy surface, its NumPy-backed direct primitives and composites, and dynamic solver callbacks                                                                                                               |
| `advect/scipy/_ndimage/`                        | Private ndimage validation, lowering, primitives, and derivative machinery behind the public module                                                                                                                               |
| `advect/xarray/`                                | Explicit registration of supported labeled containers as pytrees                                                                                                                                                                  |
| `advect/interop/`                               | First-order VJP bridges into JAX, PyTorch, and HIPS Autograd                                                                                                                                                                      |
| `packages/advect-runtime/`                      | Python-independent canonical graph artifact, `GraphStore` validation and serialization, optimization, scheduling, and staged value lifetimes                                                                                      |
| `packages/advect-native/`                       | Required PyO3 translation layer, invocation-local dynamic tape, and Python host over `advect-runtime`                                                                                                                             |
| `packages/advect/tests/`                        | Executable contract evidence, divided by the owning boundary                                                                                                                                                                      |
| `docs/`                                         | Published tutorials, concepts, compatibility statements, API reference, and this developer guide                                                                                                                                  |
| `design/`                                       | Requirements, accepted decisions, implementation status, and performance contracts                                                                                                                                                |
| `scripts/`                                      | Qualification, reporting, benchmark, release, and documentation tooling                                                                                                                                                           |

Private directories group responsibilities; their individual filenames are implementation details. Prefer locating the owning directory and reading its module docstrings over depending on a remembered leaf filename.

## Boundary rules

### Core and frontends

`advect.core` stays standard-library-only. Third-party array dependencies live in frontends or explicit adapters. Core code may define a provider-neutral hook or protocol only when at least one real core or Array API path consumes it.

NumPy protocol dispatch belongs in `advect.numpy`; Array API negotiation and revision-specific behavior belong in `core/_array_api`. Both meet at canonical operation emission. A frontend spelling, support catalog row, and canonical operation are related records, not interchangeable sources of truth.

Optional integrations are explicit. Importing `advect` installs the required NumPy frontend and built-in Array API compatibility bridge, but does not import SciPy, xarray, JAX, PyTorch, or HIPS Autograd.

### Semantics and lifetimes

Canonical operation records own stable identity and arity plus their declared abstract and derivative capabilities. Frontends and providers own concrete execution for built-in array operations; a public custom primitive's authored implementation owns its concrete call. Runtime derivative identifiers use `array.*`, `array_ext.*`, or an explicit `advect.*` internal name.

Dynamic transforms own invocation-local concrete values through the native tape. Python staging owns the immutable call/output signature and versioned `StagedProgram` envelope. `advect-runtime` owns the enclosed canonical graph artifact and `GraphStore`. Do not convert the dynamic tape through the staged optimizer or duplicate the runtime graph model in Python.

### Rust crates

`advect-runtime` has no Python, PyO3, or numerical-array dependency. It owns the canonical graph format, `GraphStore` validation, optimization, and the host-independent execution plan. Python core wraps that graph in the outer `StagedProgram` envelope and validates pytrees, call/output specifications, constant-manifest and optimization-report agreement, and other cross-record relationships. `advect-native` translates between them and owns the native dynamic tape; it must not reimplement either format, optimization, or staged scheduling.

### Sources of truth

- The canonical registry determines which built-in operations exist and which rules they provide.
- Python core's `StagedProgram` format and codecs determine the outer envelope; `advect-runtime`'s graph format determines the enclosed canonical artifact.
- Frontend support declarations determine which public spellings and lifetimes are claimed; executable support cases must cover those declarations in both directions.
- `advect.support_catalog()` derives the machine-readable public report from live declarations and registered semantics.
- The NumPy, Array API, CuPy, and SciPy pages under `docs/compatibility/` are generated projections, not another support registry. The xarray page is a hand-written structure contract because it has no callable inventory.
- `design/decisions/` records why a boundary exists. This developer guide records how to work within the current boundary.
