# Advect

A small, extensible automatic-differentiation core for Python array programs.
User code stays ordinary NumPy or Array API code; Advect owns differentiation,
functionalization, and the staged program boundary.

!!! warning "Work in Progress"

    Advect is under active development. APIs may change.

## Install

```bash
python -m pip install advect
```

The single distribution installs the core, the differentiation transforms, the
NumPy frontend, and the required Array API compatibility bridge. SciPy helpers
and xarray labels are extras: `advect[scipy]`, `advect[xarray]`, or
`advect[scientific]` for the pair.
First-order host-autodiff bridges are installed individually with
`advect[torch]`, `advect[jax]`, or `advect[autograd]`.

## Quickstart

`grad` differentiates a plain Python function — press `[ run ]` to execute it
in your browser:

```{.python .run}
import numpy as np

import advect as ad


def energy(x):
    centered = x - np.mean(x)
    return np.sum(centered**2)


x = np.linspace(0.0, 1.0, 5)
print(ad.grad(energy)(x))
```

The function is traced with its concrete inputs on every call, so ordinary
Python control flow just works. When one signature runs repeatedly — or a
program needs to be saved and loaded elsewhere — `stage` builds a durable,
serializable graph from the same code.

## Where next

- [Tutorials](tutorials/index.md) — runnable pages from the first gradient to
  custom primitives
- [Architecture](architecture.md) — how the pieces fit together and why
- [API Reference](api/index.md) — the public surface, auto-generated from
  docstrings
- The [playground](playground.md) — `[2:playground]` in the bar — traces
  expressions live and draws their derivative graphs
