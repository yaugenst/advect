# Image Processing

These are explicit differentiable counterparts to [`scipy.ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html). Use the [compatibility table](https://yaugenst.github.io/advect/0.1.0/compatibility/scipy/index.md) for exact dynamic, staged, and serialized coverage.

## ndimage

Traceable counterparts to frequently used `scipy.ndimage` operations.

### gaussian_filter

```python
gaussian_filter(
    input: object,
    sigma: object,
    order: object = 0,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    truncate: object = 4.0,
    *,
    radius: object = None,
    axes: object = None,
) -> object
```

Apply a multidimensional Gaussian filter with exact boundary adjoints.

### gaussian_filter1d

```python
gaussian_filter1d(
    input: object,
    sigma: object,
    axis: object = -1,
    order: object = 0,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    truncate: object = 4.0,
    *,
    radius: object = None,
) -> object
```

Apply a one-dimensional Gaussian filter along `axis`.

### uniform_filter

```python
uniform_filter(
    input: object,
    size: object = 3,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Apply a multidimensional uniform filter.

### uniform_filter1d

```python
uniform_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object
```

Apply a one-dimensional uniform filter along `axis`.

### convolve

```python
convolve(
    input: object,
    weights: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Multidimensional convolution with differentiable input and weights.

### correlate

```python
correlate(
    input: object,
    weights: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Multidimensional correlation with differentiable input and weights.

### convolve1d

```python
convolve1d(
    input: object,
    weights: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object
```

One-dimensional convolution with differentiable input and weights.

### correlate1d

```python
correlate1d(
    input: object,
    weights: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object
```

One-dimensional correlation with differentiable input and weights.

### laplace

```python
laplace(
    input: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    *,
    axes: object = None,
) -> object
```

Apply the discrete multidimensional Laplace operator.

### gaussian_laplace

```python
gaussian_laplace(
    input: object,
    sigma: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    *,
    axes: object = None,
    **kwargs: object,
) -> object
```

Apply a Laplacian of Gaussian filter.

### sobel

```python
sobel(
    input: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
) -> object
```

Calculate an axis-specific Sobel filter.

### prewitt

```python
prewitt(
    input: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
) -> object
```

Calculate an axis-specific Prewitt filter.

### maximum_filter

```python
maximum_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a multidimensional maximum filter with symmetric tie gradients.

### minimum_filter

```python
minimum_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a multidimensional minimum filter with symmetric tie gradients.

### maximum_filter1d

```python
maximum_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object
```

Calculate a one-dimensional maximum filter.

### minimum_filter1d

```python
minimum_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object
```

Calculate a one-dimensional minimum filter.

### grey_dilation

```python
grey_dilation(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a greyscale dilation with symmetric tie gradients.

### grey_erosion

```python
grey_erosion(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a greyscale erosion with symmetric tie gradients.

### median_filter

```python
median_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a multidimensional median filter.

### rank_filter

```python
rank_filter(
    input: object,
    rank: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a multidimensional rank filter.

### percentile_filter

```python
percentile_filter(
    input: object,
    percentile: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate a multidimensional percentile filter.

### grey_opening

```python
grey_opening(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Apply greyscale erosion followed by greyscale dilation.

### grey_closing

```python
grey_closing(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Apply greyscale dilation followed by greyscale erosion.

### morphological_gradient

```python
morphological_gradient(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate the difference between greyscale dilation and erosion.

### morphological_laplace

```python
morphological_laplace(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate the morphological Laplace operator.

### white_tophat

```python
white_tophat(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate the difference between the input and its greyscale opening.

### black_tophat

```python
black_tophat(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object
```

Calculate the difference between greyscale closing and the input.
