# NumPy Frontend

Most code can keep importing [NumPy](https://numpy.org/doc/stable/) as usual.
Its own
[ufunc and array-function protocols](https://numpy.org/doc/stable/user/basics.dispatch.html)
route traced calls through Advect. [`advect.numpy`](numpy.md#advect.numpy) is a
companion namespace for constructors that need to preserve live Advect values;
it forwards everything else to the installed NumPy module. The
[first tutorial](../tutorials/gradients.md#differentiate-a-numpy-function)
shows that path with ordinary NumPy code.

The installed NumPy minor determines exact signatures. Dynamic, staged,
serialized, and derivative support remain explicit per callable, so consult the
generated [NumPy compatibility page](../compatibility/numpy.md) rather than
inferring support from attribute availability.

Traced and staged `numpy.linalg` decompositions keep NumPy's field names:
`eig`, `eigh`, `qr`, `svd`, and `slogdet` return named tuples such as
`EighResult`. These are Advect's result types rather than `numpy.linalg`'s own
classes, and a [`vjp`](transforms.md#advect.vjp) or
[`vjp_program`](staging.md#advect.vjp_program) cotangent for such an output
must have the same container type. Build it from the output, for example with
[`tree_map`](pytree.md#advect.pytree.tree_map)`(np.ones_like, output)` or
`type(output)(...)`; a plain tuple is rejected as a structure mismatch. The
[host-framework bridges](interop/index.md) place host cotangents in this
structure themselves.

::: advect.numpy
