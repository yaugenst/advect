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

::: advect.numpy
