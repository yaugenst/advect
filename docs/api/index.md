# API Reference

Most work begins with functions imported directly from `advect`. Optional
NumPy, SciPy, xarray, and host-framework integrations live in their own
modules. The [tutorials](../tutorials/index.md) teach complete workflows; these
pages collect the installed signatures and exact API contracts.

| Public API | Responsibility |
| --- | --- |
| [Transforms](transforms.md) | Dynamic differentiation, higher-order transforms, checkpointing, and implicit roots |
| [Staging](staging.md) | Immutable programs, serialization, and staged differentiation |
| [Primitives](primitives.md) | Public custom-operation authoring |
| [Arrays](arrays.md) | Provider-preserving construction and tracer boundaries |
| [Pytrees](pytree.md) | Structured inputs, outputs, and custom node registration |
| [Testing utilities](testing.md) | Numerical checks for composed functions and custom primitives |
| [Support catalog](support.md) | Machine-readable operation and lifetime claims |
| [Errors](errors.md) | Public diagnostics and exception hierarchy |
| [NumPy frontend](numpy.md) | Explicit constructors and transparent NumPy namespace behavior |
| [SciPy](scipy/index.md) | Optional special functions, image processing, and solver callbacks |
| [xarray](xarray.md) | Optional labeled-container pytree registration |
| [Host autodiff interop](interop/index.md) | Optional JAX, PyTorch, and HIPS Autograd VJP bridges |

## Shared semantics

Dynamic [transforms](transforms.md) trace concrete values for each call and
preserve the Python control flow that ran. [`stage`](staging.md#advect.stage)
instead compiles one shape-and-dtype signature into an immutable graph; staged
and serialized support are therefore separate claims from dynamic support.

Python scalars promote weakly, as in NumPy 2 (NEP 50), in every lifetime:
eager, traced, staged, and a staged program called inside another trace or
stage. A value is weak exactly where eager Python holds a Python scalar: a
Python scalar input, or a Python operator (arithmetic, comparison, bitwise,
`abs`, `.real`, `.imag`) applied only to weak values; augmented assignment such
as `s += 1.0` rebinds a Python scalar, as in Python. The `real` and `imag`
functions read those attributes, and `diff` with `n=0` returns its input, so
they keep a weak value weak, as NumPy's do.
Other NumPy and Array API functions and array methods return strong values even
when every argument is weak, so `np.sin(s) * x32` and `xp.multiply(s, s) * x32`
are `float64` while `(s * s) * x32` stays `float32`. Advect's
[`array`](arrays.md#advect.array) and [`asarray`](arrays.md#advect.asarray) are
strong too, as NumPy's are, while
[`stop_gradient`](arrays.md#advect.stop_gradient) keeps a weak value weak, as
the eager identity does. A Python operator on weak values computes exactly as
Python does, so `1.0 / (s - s)` raises `ZeroDivisionError`; a staged program
keeps the dtype it declared and rejects a negative weak base raised to a
fractional power, which Python makes complex.

Selected real Python scalars are lifted to zero-dimensional `float64` arrays,
which also provide the array methods a Python float lacks, such as `copy`.
Derivatives with respect to them return as Python scalars, as does a weak
output. Unlike eager NumPy, `item()` returns a strong rank-zero value so that
its derivative stays attached.

Structured inputs and outputs use [pytrees](pytree.md). Complex
differentiation is real-linear; use [`jvp`](transforms.md#advect.jvp),
[`vjp`](transforms.md#advect.vjp), or
[`linearize`](transforms.md#advect.linearize) when the output is complex.

Importing `advect` is enough for NumPy and Array API code. SciPy, xarray, and
host-framework integrations use their own optional imports. The generated
[compatibility catalog](../compatibility/index.md) lists which calls can run
dynamically, be staged, be saved, and be differentiated.
