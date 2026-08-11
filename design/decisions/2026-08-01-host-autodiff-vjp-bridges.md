# ADR: Host Autodiff VJP Bridges

**Date:** 2026-08-01
**Status:** Accepted

## Context

JAX, PyTorch, and HIPS Autograd already own tracing, array semantics, and
derivative execution. Treating their arrays as ordinary Advect providers would
create nested tapes and make it unclear which framework owns transformation,
device placement, and gradient conventions. The concrete need is narrower:
use a NumPy-backed Advect callable as one differentiable operation inside a
host framework program.

The boundary must also preserve Advect's real-adjoint convention. PyTorch uses
the same complex cotangent representation at a custom reverse rule. JAX and
HIPS Autograd use the conjugate, complex-bilinear representation, so both the
incoming output cotangent and returned input gradient need conversion.

## Decision

The single `advect` distribution contains three explicitly imported modules:

- `advect.interop.torch`, enabled by `advect[torch]`;
- `advect.interop.jax`, enabled by `advect[jax]`;
- `advect.interop.autograd`, enabled by `advect[autograd]`.

There is no aggregate dependency extra: users install only the host framework
they use. Importing `advect` or `advect.interop` imports none of them.

Each framework module exports one `wrap` function. A wrapped callable accepts
one or more positional or keyword tuple, list, or dictionary pytrees whose
leaves are standard NumPy floating or complex values; every supplied leaf is
differentiable. A custom container must be recognized by both Advect and its
host. Static configuration is closed over by the callable. Differentiable
outputs are nonempty pytrees with the same dtype boundary. The bridge is
first-order reverse mode only and does not register an array provider, change
Advect's resolver, enter durable staging, or expose a second derivative engine.

The framework-specific ownership is:

- PyTorch uses one eager `torch.autograd.Function`. Its forward copies inputs
  to host NumPy, retains one Advect pullback, and returns outputs to the common
  input device. One backward consumes that pullback. PyTorch cotangents pass
  through without conjugation.
- HIPS Autograd uses one custom primitive per traced invocation. The primitive
  retains the matching reusable Advect linearization, applies its pullback for
  each host cotangent, and conjugates complex cotangents and gradients at the
  boundary. The host VJP closure owns and closes the linearization. A nested
  Autograd trace rejects explicitly.
- JAX uses `custom_vjp`. Concrete eager calls execute the NumPy-backed callable
  directly and infer its output. If JIT compilation or abstract shape
  evaluation begins without an explicit output shape/dtype pytree, the bridge
  raises an actionable error. Supplying that pytree enables `pure_callback` for
  `jit` and `eval_shape`, but does not define `vmap` batching semantics.
  `has_aux=True` interprets the callable's result as `(value, aux)` and excludes
  the JAX-compatible auxiliary pytree from the Advect VJP.
  Both backward paths replay the pure callable at the saved primal values,
  build an Advect VJP, consume it immediately, and conjugate complex
  cotangents and gradients. Forward- and higher-order transforms remain
  excluded.

## Consequences

Host applications can compose Advect derivatives without making JAX or
PyTorch an Advect provider. The base install and import path stay small, and
each optional dependency has an independent extra and qualification lane.

The adapters cross a host NumPy boundary; they are not device-native kernels.
PyTorch performs an explicit host round trip. JAX may schedule or duplicate a
staged pure callback and replays the forward computation during reverse mode.
A remote or otherwise effectful operation needs an application-owned adapter
with an explicit residual/token lifecycle rather than the generic JAX bridge.

The initial contract intentionally excludes static leaf annotations,
`torch.func`, `torch.compile`, JAX `jvp`, `vmap`, higher derivatives, and
durable artifacts. A concrete application need may extend one of these
boundaries without turning the frameworks into providers.
