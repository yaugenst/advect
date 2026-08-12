# NumPy Frontend

Most code can keep importing [NumPy](https://numpy.org/doc/stable/) as usual. Its own [ufunc and array-function protocols](https://numpy.org/doc/stable/user/basics.dispatch.html) route traced calls through Advect. [`advect.numpy`](https://yaugenst.github.io/advect/0.1.1/api/numpy/#advect.numpy) is a companion namespace for constructors that need to preserve live Advect values; it forwards everything else to the installed NumPy module. The [first tutorial](https://yaugenst.github.io/advect/0.1.1/tutorials/gradients/#differentiate-a-numpy-function) shows that path with ordinary NumPy code.

The installed NumPy minor determines exact signatures. Dynamic, staged, serialized, and derivative support remain explicit per callable, so consult the generated [NumPy compatibility page](https://yaugenst.github.io/advect/0.1.1/compatibility/numpy/index.md) rather than inferring support from attribute availability.

## numpy

NumPy dispatch, tracing, and a transparent compatibility namespace.

This package contains the NumPy-specific implementation:

- TracedArray: Wrapper that intercepts NumPy operations
- Ufunc dispatch: Handling for np.add, np.sin, etc.
- Array function dispatch: Handling for np.sum, np.reshape, etc.

Ordinary attributes are returned directly from NumPy. Only constructors that need to preserve live Advect values are defined here.

### array

```python
array(
    object: object,
    dtype: object | None = None,
    *,
    copy: bool | None = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
    like: object | None = None,
) -> Any
```

Create a NumPy array while preserving live values selected by `like=`.

### asanyarray

```python
asanyarray(
    a: object,
    dtype: object | None = None,
    order: str | None = None,
    *,
    device: object | None = None,
    copy: bool | None = None,
    like: object | None = None,
) -> Any
```

Convert to a NumPy array or subclass without detaching live Advect values.

### asarray

```python
asarray(
    a: object,
    dtype: object | None = None,
    order: str | None = None,
    *,
    device: object | None = None,
    copy: bool | None = None,
    like: object | None = None,
) -> Any
```

Convert to a base NumPy array without detaching live Advect values.

### __getattr__

```python
__getattr__(name: str) -> Any
```

Return unmodified attributes from the installed NumPy module.

### __dir__

```python
__dir__() -> list[str]
```

Expose the real NumPy namespace plus Advect's explicit overrides.
