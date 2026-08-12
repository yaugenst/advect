# Special Functions

Use these explicit Advect counterparts when a [`scipy.special`](https://docs.scipy.org/doc/scipy/reference/special.html) function must remain differentiable or stageable. See the [Scientific Python tutorial](https://yaugenst.github.io/advect/0.1.1/tutorials/scientific-python/#use-the-differentiable-scipy-namespace) for a `logsumexp` gradient and the [compatibility table](https://yaugenst.github.io/advect/0.1.1/compatibility/scipy/index.md) for exact coverage.

## special

Traceable high-value counterparts to `scipy.special`.

### gammaln

```python
gammaln(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the logarithm of the absolute gamma function.

### digamma

```python
digamma(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the logarithmic derivative of the gamma function.

### polygamma

```python
polygamma(n: object, x: object) -> object
```

Compute the `n`-th derivative of `digamma` with SciPy broadcasting.

### erf

```python
erf(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the error function.

### erfc

```python
erfc(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the complementary error function.

### erfcx

```python
erfcx(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the scaled complementary error function.

### erfinv

```python
erfinv(
    y: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the inverse error function.

### expit

```python
expit(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the logistic sigmoid.

### log_expit

```python
log_expit(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the logarithm of the logistic sigmoid.

### ndtr

```python
ndtr(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the standard normal cumulative distribution function.

### log_ndtr

```python
log_ndtr(
    x: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the logarithm of the standard normal cumulative distribution.

### ndtri

```python
ndtri(
    p: object, /, out: object = None, **kwargs: object
) -> object
```

Compute the inverse standard normal cumulative distribution.

### logsumexp

```python
logsumexp(
    a: object,
    axis: object = None,
    b: object = None,
    keepdims: bool = False,
    return_sign: bool = False,
) -> object
```

Compute SciPy-compatible weighted, optionally signed log-sum-exp.

### softmax

```python
softmax(x: object, axis: object = None) -> object
```

Compute the softmax function along `axis`.

### log_softmax

```python
log_softmax(x: object, axis: object = None) -> object
```

Compute the logarithm of the softmax function along `axis`.
