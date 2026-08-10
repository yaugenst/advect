# NumPy Frontend

[`advect.numpy`](numpy.md#advect.numpy) is a transparent companion namespace for
ordinary [NumPy](https://numpy.org/doc/stable/) code. It overrides only
constructors that must preserve live Advect values and forwards other
attributes to the installed NumPy module. NumPy's own
[ufunc and array-function protocols](https://numpy.org/doc/stable/user/basics.dispatch.html)
perform tracing; users do not register operations in this namespace. The
[first tutorial](../tutorials/gradients.md#differentiate-a-numpy-function)
shows that path with ordinary NumPy code.

The installed NumPy minor determines exact signatures. Dynamic, staged,
serialized, and derivative support remain explicit per callable, so consult the
generated [NumPy compatibility page](../compatibility/numpy.md) rather than
inferring support from attribute availability.

::: advect.numpy
