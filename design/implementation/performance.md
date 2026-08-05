# Runtime and Performance

Advect has two execution lifetimes and three distinct evidence products.

| Product | Purpose | Gate |
| --- | --- | :---: |
| Advect regression | Detect a candidate regression against an exact reference Advect artifact | yes |
| Runtime memory | Establish named lifetime and memory invariants in isolated workers | yes |
| Ecosystem comparison | Give historical context against HIPS Autograd and future peer systems | no |

Provider qualification is correctness evidence. It does not publish launch-heavy
microtimings or allocator snapshots; provider performance belongs in a named
benchmark when a requirement makes it consequential.

## Advect regression acceptance

The blocking performance lane compares two wheels:

- an exact reference Advect wheel and source revision;
- an exact candidate Advect wheel and source revision.

The harness hashes both wheels and runs each in an isolated, non-project `uv`
worker on the same host. Reference/candidate order alternates by replicate. The
report rejects interpreter, platform, NumPy, diagnostic-mode, or correctness
differences before treating timings as comparable. Both artifacts must use a
release native build. Their complete `advect_native` records are not required to
be identical: native implementation details may legitimately differ between the
reference and candidate being measured.

Dynamic workloads report these public lifecycle phases independently:

1. a one-shot gradient call, including its concrete trace and release;
2. `linearize` plus the required `LinearMap.close()`;
3. repeated reverse application from one retained linearization.

The same stencil supplies the staged lane. It reports gradient compilation,
warm gradient execution, and serialized round-trip plus execution. Its
correctness preflight compares dynamic, staged, and restored gradients and
verifies that the input is unchanged. These six phases cover the runtime
lifetime boundaries without maintaining a synthetic workload matrix.

Acceptance is per workload and phase. For each paired replicate the harness
computes `candidate / reference`, then derives a threshold from the relative
median absolute deviation of those ratios:

```text
threshold = max(configured minimum, noise multiplier * paired relative MAD)
```

Evidence is unstable when that derived threshold exceeds the configured
ceiling. The default minimum is 5%, the multiplier is 6, and the stability
ceiling is 20%. These values are explicit command inputs and recorded in the
JSON together with size, warmup, rounds, block size, warmed replicate count,
alternating order, raw samples, exact wheel paths, byte sizes, hashes, and source
revisions. Regression and worker reports use schema version 2; schema-1 reports
remain historical evidence and have no compatibility reader. A geometric mean
cannot hide a regression in one workload or lifecycle.

Run the gate after building both wheels:

```bash
uv run python -m scripts.bench_advect_regression \
  --reference-wheel artifacts/wheels/reference/advect-*.whl \
  --candidate-wheel artifacts/wheels/candidate/advect-*.whl \
  --reference-revision "$(git rev-parse <reference>)" \
  --candidate-revision "$(git rev-parse <candidate>)" \
  --warmed-replicates 5 \
  --acceptance \
  --format json \
  --output artifacts/benchmarks/advect-regression.json
```

The wildcard above is illustrative; pass one resolved wheel path to the
harness. Acceptance requires at least five warmed replicates and release native
extensions with shipping diagnostics.

## Ecosystem comparison

The HIPS Autograd report remains useful historical context:

```bash
uv run --with autograd python -m scripts.bench_autodiff_runtime \
  --suite all \
  --warmed-replicates 3 \
  --phases \
  --format json \
  --output artifacts/benchmarks/dynamic-ecosystem-comparison.json
```

It validates results, warms both implementations, disables GC while timing,
and alternates Advect/HIPS blocks. Replicates share one process and are named
accordingly. The report is explicitly non-gating. It contains no same-process
memory measurement; isolated runtime-memory workers are the memory authority.

A future Advect/HIPS/PyTorch/JAX table remains a separate non-gating product.
It must distinguish eager from compiled execution, compile cost from steady
state, primal from derivative work, and CPU from GPU rather than collapsing
those contracts into one rank.

## Memory acceptance

Memory measurement uses one child process per case and run. Profiling workers
and timing workers are separate. Every acceptance scenario first runs a small
deterministic correctness preflight in another worker, so reference allocations
cannot contaminate the measured process.

Acceptance requires one exact profile:

| Profile | Exact scope | Gated invariants |
| --- | --- | --- |
| `cpu-runtime` | allocation calibration, elementwise, stencil, checkpoint plain/checkpoint pair, residual, reusable linear map, captured staged constant | stability; release native build; checkpoint memory/runtime tradeoff and replay count; residual release exactly once; retained linear-map state followed by zero post-close ownership; attributed staged constant cache |
| `cupy-donation` | allocation calibration and donation/forced-fresh functional updates | stability; release native build; one destination buffer avoided; provider high-water and runtime ratios; zero post-close ownership |

Both profiles require exactly five successful memory runs for every case. They
run exactly five successful timing workers, with exactly ten calls each, only
for the control pair whose
runtime enters the verdict: checkpoint/plain on CPU and donation/forced-fresh
on CuPy. Acceptance also requires a 64 MiB live-data target, an allocation-probe
calibration within 10%, and median-absolute variation no greater than 5% for
the measurements it consumes. Reported fields outside the profile's named
checks are diagnostic; their presence does not imply acceptance.

Run CPU acceptance:

```bash
ADVECT_SOURCE_REVISION="$(git rev-parse HEAD)" \
uv run python -m scripts.bench_runtime_memory \
  --profile cpu-runtime \
  --acceptance \
  --format json \
  --output artifacts/benchmarks/runtime-memory-cpu.json
```

Run the separately qualified GPU profile on a CUDA host:

```bash
ADVECT_SOURCE_REVISION="$(git rev-parse HEAD)" \
uv run python -m scripts.bench_runtime_memory \
  --profile cupy-donation \
  --acceptance \
  --format json \
  --output artifacts/benchmarks/runtime-memory-cupy-donation.json
```

Every controller run requires a named `--profile`; `--smoke` reduces that
profile to one small run per case without changing its case matrix.

## Adding performance evidence

Adding a supported function normally requires conformance evidence, not a new
permanent microbenchmark.

| Change | Required performance evidence |
| --- | --- |
| ordinary alias, composite, or primitive without a performance claim | none; run conformance |
| new runtime mechanism or lifetime behavior | extend one representative end-to-end scenario |
| bespoke VJP retained because it is faster | compare it with structural transposition at a representative shape |
| change to a known hot path | run the existing reference/candidate gate |
| new provider execution class with a performance claim | add one synchronized provider scenario |
| refactor with no expected performance effect | run the existing gate; add no new case |

The permanent regression gate stays deliberately singular. Add a phase only
when a new runtime boundary cannot be measured by an existing phase. Add a new
workload only when a concrete performance requirement cannot be represented by
the stencil; record that requirement beside the workload when it is added.

## Evidence and optimization boundaries

Benchmark and qualification reports have a `schema_version`, `report_kind`,
source revision, and environment provenance. Regression reports additionally
record exact wheel paths, SHA-256 digests, and alternating process order. Raw
JSON belongs in CI or release artifacts under `artifacts/`; it is not runtime
source and does not justify a permanent universal claim.

Numerical kernels remain provider-owned. Dynamic tracing must not acquire
durable hashing, graph optimization, or cache machinery to improve a staged
number. The staged executor may donate only an internally owned, last-use,
unaliasable compatible temporary. Logical SSA values and caller inputs remain
unchanged.
