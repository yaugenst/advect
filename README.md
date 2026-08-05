# Advect

Advect differentiates ordinary NumPy programs and can turn the same code into
an immutable, serializable array program. Dynamic transforms preserve Python
control flow; explicit staging creates the durable graph boundary.

Advect is alpha software. Public APIs and serialized artifacts may change
before 1.0.

## Install

Advect requires Python 3.12 or 3.13.

```bash
python -m pip install advect
```

SciPy helpers and xarray-aware gradients are available together:

```bash
python -m pip install "advect[scientific]"
```

The first public release is still being prepared. Until it is published, the
same package can be installed from a checkout with `python -m pip install .`.

## First gradient

`grad` traces a concrete call and returns a function with the same input
structure. The code being differentiated stays ordinary NumPy:

```python
import numpy as np

import advect as ad


def energy(x):
    centered = x - np.mean(x)
    return np.sum(np.sin(centered) ** 2)


x = np.linspace(0.0, 1.0, 8)
gradient = ad.grad(energy)(x)
```

Each call follows the Python control flow taken by its concrete inputs. Local
array mutation is functionalized while tracing, including ordinary
accumulation and basic indexed updates. Mutating inputs or mutating through an
ambiguous view fails at the operation that crosses the boundary.

## Dynamic calls and staged programs

Use dynamic transforms for ordinary Python execution. Use `stage` when one
shape and dtype signature must run repeatedly, move between processes, or be
saved:

```python
program = ad.stage(energy, x)
result = program(x)

payload = program.to_dict()
restored = ad.StagedProgram.from_dict(payload)
```

A staged program is immutable and records its required Array API revision.
Staged differentiation returns durable programs as well: `ad.grad(program)`
builds a scalar gradient, while `ad.vjp_program(program)` builds a reusable
pullback with an explicit cotangent input.

Complex derivatives are real-linear. For a real loss,
`grad(abs(z) ** 2) == 2 * z`; complex-output `grad` is rejected in favor of
`linearize`, `jvp`, or `vjp`.

## Providers and scientific Python

NumPy 2.0 through 2.4 is the primary user path and a first-class frontend.
Advect also carries explicit Array API 2022.12, 2023.12, and 2024.12 contracts
for provider-neutral execution and serialized programs. The
[`support_catalog`](https://yaugenst.github.io/advect/latest/compatibility/)
reports dynamic, staged, and serialized support separately; the presence of a
registered operation is not itself a support claim.

CuPy uses the built-in `array-api-compat` bridge. A clean-worktree local gate
has passed all three supported Array API revisions with CuPy 14.1.1, including
the donation memory profile. No immutable public release artifact carries that
qualification yet, so it is evidence for the release candidate rather than a
published compatibility promise.

The optional `advect.scipy` namespace provides a bounded NumPy-backed set of
differentiable special functions, image filters, and root/GMRES solver
adapters. Importing `advect.xarray` registers floating- and complex-valued
`DataArray` and `Dataset` pytrees whose gradients preserve labels and metadata.

Optional bridges make a NumPy-backed Advect function one differentiable
operation inside PyTorch, JAX, or HIPS Autograd:

```bash
python -m pip install "advect[torch]"  # or advect[jax], advect[autograd]
```

These are first-order host-transform boundaries, not additional array
providers. The
[interop reference](https://yaugenst.github.io/advect/latest/api/interop/)
defines their eager, compiled, pytree, and auxiliary-output contracts.

## Extending Advect

A custom operation starts with its concrete implementation and adds only the
semantic rules it needs:

```python
@ad.primitive
def solve(a, b): ...


@solve.def_abstract
def solve_abstract(a, b): ...


@solve.def_jvp
def solve_jvp(output, primals, tangents): ...
```

Advect infers a default operation name from the function. Package authors may
pass `name="acme.solve"` when serialized programs need an identity independent
of the Python module path. The name is a link key, not automatic semantic
versioning: a serialized artifact must be loaded with its matching operation
implementation.

## Documentation and development

The [documentation](https://yaugenst.github.io/advect/) starts with runnable
tutorials, then separates architecture, API, and compatibility contracts.
[CONTRIBUTING.md](https://github.com/yaugenst/advect/blob/main/CONTRIBUTING.md)
contains the setup and direct checks for each kind of change. Security reports
belong in the repository's
[private advisory form](https://github.com/yaugenst/advect/security/advisories/new),
not a public issue.

Advect is licensed under the
[MIT License](https://github.com/yaugenst/advect/blob/main/LICENSE).
