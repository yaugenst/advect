# Array Provider Qualification

**Status:** Implemented

## Outcome

Advect's backend-neutral transform path is automatically qualified against
Array API 2022.12, 2023.12, and 2024.12 on `array-api-strict`, plus the
corresponding NumPy 2.0-2.5 profiles. One representative scientific program
uses a shared source body, complex64 FFTs, a complex linear solve, a real
scalar loss, three differentiated arguments, and a Python weak scalar. The
qualification covers:

- dynamic `value_and_grad`, JVP, and VJP;
- an abstractly staged primal and staged derivative;
- derivative artifact serialization and reload;
- a captured NumPy constant materialized by the runtime provider;
- copied-field basic-slice mutation in dynamic and staged gradients;
- dtype, concrete array type, and device preservation;
- a finite-difference directional check and cross-provider numerical agreement.

The same serialized derivative graph executes on both automated providers for
each selected revision. A clean-worktree pre-release run also qualifies all
three profiles with CuPy 14.1.1 from an exact wheel and passes the
`cupy-donation` acceptance profile. Its private release-preparation record
retains the wheel, source revision, device, driver, CUDA runtime, reports, and
digests. Publishing a provider claim still requires running that gate for the
final candidate and uploading the complete evidence as an immutable release or
workflow artifact.

## Evidence scope

The ordinary test suite and manual `Numerical Evidence` workflow exercise the
NumPy and `array-api-strict` CPU paths for all three revisions. The local GPU
lane uses an isolated Python 3.12 environment on one NVIDIA GeForce RTX 4080
SUPER with driver 610.43.03:

| CuPy | Distribution | CUDA runtime | Array API profiles | Evidence status |
| --- | --- | --- | --- | --- |
| 14.1.1 | `cupy-cuda13x[ctk]` | 13.2 | 2022.12, 2023.12, 2024.12 | Local pre-release gate passed; no public immutable artifact |

The commands below reproduce the gate on a suitably provisioned host. Generated
reports remain ignored by Git and must be uploaded as release or workflow
artifacts before the public compatibility catalog can call the provider pass
published evidence.

## Supported qualification matrix

| Path | NumPy 2.0-2.5 | `array-api-strict` 2022.12-2024.12 | CuPy through compat |
| --- | --- | --- | --- |
| Dynamic value and multi-argument gradient | Pass | Pass | Local pass |
| Dynamic JVP and VJP | Pass | Pass | Local pass |
| Staged primal and derivative | Pass | Pass | Local pass |
| Serialized staged derivative | Pass | Pass | Local pass |
| Complex64 FFT and `linalg.solve` | Pass | Pass | Local pass |
| `1j * float32 -> complex64` | Pass | Pass | Local pass |
| Captured host constant | CPU | CPU | Local device-resident pass |
| Copied-field basic-slice mutation | Pass | Pass | Local pass |
| Provider type, dtype, and device preserved | Pass | Pass | Local pass |

NumPy uses its first-class Advect frontend. `array-api-strict` exercises the
generic Array API frontend directly. CuPy uses the base `array-api-compat`
dependency through Advect's fixed private fallback, configured when `advect` is
imported. The supported source style obtains a namespace from the traced value.
This means `ad.grad(f)(provider_array)` and
`ad.stage(f, specs=...)` work without calls to `cupy.*` inside `f`. A plain
call to the same function on a raw provider array still depends on whether it
implements
`__array_namespace__` itself.

The fallback is not hard-coded to CuPy, but GPU qualification is. Providers with
their own autodiff system do not justify another support and testing matrix
without a concrete interoperability use case. An unqualified provider may work
through `array-api-compat`; that is incidental compatibility, not an Advect
contract. Arrays explicitly attached to another autodiff tape are rejected
rather than silently composing two transform systems. CuPy remains the
designated donation provider. Its qualification and donation reports form one
configuration-level evidence set that must stay attached to the exact wheel
and source revision being claimed.

The qualification script records correctness summaries plus graph size and
optimization counts. Provider timing and allocator state belong to named
performance and memory profiles rather than this correctness report.

## Admitted extension surface

The staged FFT surface is:

```text
fft, ifft, fftn, ifftn
rfft, irfft, rfftn, irfftn
fftshift, ifftshift
```

The staged linear-algebra surface qualified here is `linalg.solve`. Frequency
constructors have no traced array operand and remain ordinary provider calls.
The generated inventory below records additional declared dynamic and staged
linear-algebra capability. Those functions remain outside this
execution-qualification claim until they are selected by the upstream and
provider gates. `hfft`, `ihfft`, and extension functions classified as missing
or unsupported in that inventory are not admitted.

## Official Array API trace-and-execute qualification

The repository includes a bridge for the official `array-api-tests` suite.
Array creation and assertion helpers continue to use `array-api-strict`.
Every callable selected by the executable qualification cases is reconstructed
inside an Advect concrete trace, an Advect staged graph, or a
serialized-and-restored staged graph when the official suite invokes it. Other
calls pass directly to the reference provider so the whole upstream suite can
run without treating qualification coverage as the user-facing support list.
The bridge logs transformed calls and reports selected operations not observed
by the upstream suite; deterministic operation cases cover zero-array-input and
otherwise unobserved forms separately.

The pinned suite revision is
`5d0b701b0c4ab6ec98794068cf7af393a8a51c61`. The selection comes from the
executable qualification catalog rather than a second list in the runner. Long
runs may be split deterministically with `--shard-count` and `--shard-index`;
a complete evidence set contains every shard for every selected mode and Array
API revision.

Reproduce the gate with:

```bash
git clone https://github.com/data-apis/array-api-tests /tmp/array-api-tests
git -C /tmp/array-api-tests checkout \
  5d0b701b0c4ab6ec98794068cf7af393a8a51c61
git -C /tmp/array-api-tests submodule update --init

for version in 2022.12 2023.12 2024.12; do
  uv run --with 'hypothesis>=6.151.0' --with ndindex \
    --with pytest-json-report \
    python -m scripts.run_array_api_conformance \
    --suite-path /tmp/array-api-tests \
    --array-api-version "$version" \
    --mode all \
    --max-examples 10 \
    --output "artifacts/qualification/$version/array-api-official-suite.json"
done
```

`array-api-strict` remains the reference provider for the raw standard. Passing
an untransformed unsupported test is provider evidence, not an Advect support
claim; only logged transformed calls and the selected deterministic cases count
as Advect evidence.

Each shard first runs directly against `array-api-strict`; only nodes that pass
that baseline qualify Advect. The report retains every baseline failure and skip
by node ID. Both runs seed Python sampling from the node ID so upstream tests
that use `strategy.example()` receive the same draw instead of producing false
provider-versus-Advect differences.

## Executable qualification and structural diagnostics

The public [compatibility catalog](../../docs/compatibility/index.md) discovers Array
API candidates from the generated namespace bindings and reports dynamic,
staged, and serialized support separately. The pinned official stubs define the
upstream inventory, while executable cases qualify parameters, differentiable
operands, important value regimes, and program lifetimes.
`array-api-strict==2.4.1` remains the reference execution provider.

Named multi-output linear-algebra results retain their standard fields
dynamically and after artifact serialization. Non-default `diff` boundaries,
`searchsorted(sorter=...)`, positional `roll` shift, and `asarray` copy/device
forms have dedicated round-trip cases. Generated Hypothesis tests select live
differentiable parameters independently. For every complete differentiable
baseline case in each revision, the gate runs dynamic JVP and VJP plus staged,
serialized VJP execution and checks the JVP/VJP adjoint identity. Primitive
conformance independently anchors the underlying numerical rule to finite
differences.

The older generated registry report remains useful as a structural diagnostic:
it joins the reference namespace to binder, abstract-rule, and derivative
registries. Its labels such as `staged` mean that one registered invocation can
stage, not that the whole upstream callable contract is complete. It therefore
does not independently define support.

Deterministic execution cases back every advertised staged invocation on
`array-api-strict`; a portable scientific subset also runs on NumPy. These
cases complement the upstream Hypothesis suite by covering fixed result
structures, creation calls, and material static options directly.

The official-suite bridge reconstructs array-returning calls inside Advect. It
does not reconstruct `can_cast`, `finfo`, `iinfo`, `isdtype`, or `result_type`,
because those are compile-time metadata forms rather than array results. The
conformance report names all five as `metadata_qualified_elsewhere`; focused
tests exercise each through dynamic tracing, staging, and serialized replay.

Generate execution evidence with:

```bash
for version in 2022.12 2023.12 2024.12; do
  uv run python -m scripts.qualify_array_api_operations \
    --array-api-version "$version" \
    --output "artifacts/qualification/$version/array-api-operations.json"

  uv run python -m scripts.qualify_array_api_operations \
    --array-api-version "$version" \
    --providers array-api-strict,numpy \
    --subset portable \
    --output "artifacts/qualification/$version/array-api-portable.json"
done
```

Generate human-readable output with:

```bash
uv run python -m scripts.report_array_api_support \
  --array-api-version 2023.12
```

Generate JSON evidence with:

```bash
uv run python -m scripts.report_array_api_support \
  --array-api-version 2023.12 \
  --format json \
  --output artifacts/qualification/2023.12/array-api-support.json
```

Generated output is ignored by Git and uploaded by the manual
`Numerical Evidence` workflow or release process. The installed
`array-api-strict` namespace defines the official versioned surface; Advect's
live registries define Advect capability. This inventory reports declared
capability. Operation reports record broad deterministic execution; the
smaller official trace-and-execute gate above remains the upstream-suite
qualification.

## Scientific provider gate

The provider matrix is executable independently:

```bash
for version in 2022.12 2023.12 2024.12; do
  uv run python -m scripts.qualify_array_providers \
    --array-api-version "$version" \
    --providers numpy,array-api-strict \
    --output "artifacts/qualification/$version/array-provider-scientific.json"
done
```

To refresh one manual CUDA 13 configuration, first create or activate an
isolated project environment, then set `cupy_version` to the version being
qualified. Repeat the whole gate for each configuration being claimed:

```bash
cupy_version=14.1.1
uv pip install "cupy-cuda13x[ctk]==$cupy_version"

uv run pytest \
  packages/advect/tests/advect_array_api_compat_tests/test_cupy_qualification.py

for version in 2022.12 2023.12 2024.12; do
  uv run python -m scripts.qualify_array_providers \
    --array-api-version "$version" \
    --providers numpy,array-api-strict,cupy \
    --output \
      "artifacts/qualification/cupy-$cupy_version/$version/array-provider-scientific.json"
done

uv run python -m scripts.bench_runtime_memory \
  --profile cupy-donation \
  --byte-budget 64MiB \
  --max-bytes 512MiB \
  --runs 5 \
  --timing-runs 5 \
  --timing-iterations 10 \
  --acceptance \
  --format json \
  --output \
    "artifacts/benchmarks/cupy-$cupy_version/runtime-memory-cupy-donation.json"
```

CuPy toolkit discovery is environment-specific. Installing the toolkit extras
into the project environment, as above, lets CuPy's pathfinder see its headers
and libraries. The generated provider report records correctness summaries plus
compile and graph sizes. The separate named memory profile owns synchronized
timing and isolated-process allocator evidence under the
[performance contract](performance.md).

## Bugs closed by qualification

The provider and upstream gates exposed concrete semantic gaps that smaller
examples did not:

- nested `fft` and `linalg` namespaces were compared with their submodule
  instead of the root provider namespace;
- NumPy traced arrays did not expose `__array_namespace__`;
- staged `linalg.solve` and FFT forms lacked complete abstract semantics;
- float64 real FFTs were declared as complex64;
- integer `sum` and `prod` missed Array API accumulation dtypes;
- `concat(axis=None)` did not model flattening;
- staged dtype attributes were not resolved through the runtime provider;
- provider arrays without `.copy()` could not enter functionalized updates;
- reverse scalar seeds and replayed weak complex scalars could move or widen;
- low-precision and weak-scalar replay needed explicit dtype preservation;
- the 2022.12 profile's floating reductions needed their historical
  accumulation dtype during generic-provider replay;
- weak Python scalars needed provider-neutral materialization before concrete
  execution, including providers implementing pre-2024 scalar rules;
- provider-specific parameter names could partially match the standard and
  drop another array operand, so binding now uses only the frozen profile
  signature.

Each correction lives at the shared frontend, abstract, evaluator, or
derivative boundary and is covered by a focused regression test.

## Deliberate boundary

The qualification is one process and one provider device. Mixed-provider
calls, multi-device transforms, arbitrary `cupy.*` source code, complete
mutation-through-view semantics, data-dependent staged shapes, and functions
classified as unsupported in the generated inventory remain unsupported.
Checkpoint replay and residual-bearing custom primitives are separate
transform contracts, not claims made by this provider matrix. Declared
extension functions outside the selected matrix are not provider-qualified
here. Adding another provider means running this same contract and fixing
shared semantics; it does not mean adding provider-specific derivative rules.
