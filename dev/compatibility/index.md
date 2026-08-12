# Compatibility

The NumPy, Array API, and SciPy tables are generated from Advect's executable support declarations. The CuPy and xarray pages state their integration boundaries separately.

Every listed function works in dynamic transforms. **Stage/save** says whether the same call can appear in a staged and serialized program. **Differentiate** reports user-visible derivative support. **No** means no derivative rule is available; **n/a** marks a structural or mathematically nondifferentiable operation.

| Integration                                                                         | Contract                                                  |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [NumPy](https://yaugenst.github.io/advect/dev/compatibility/numpy/index.md)         | 392 callable forms; NumPy 2.0-2.5 (2.5.2 in this build)   |
| [Array API](https://yaugenst.github.io/advect/dev/compatibility/array-api/index.md) | 169 namespace functions across revisions 2022.12-2024.12  |
| [CuPy](https://yaugenst.github.io/advect/dev/compatibility/cupy/index.md)           | Single-device Array API provider path                     |
| [SciPy](https://yaugenst.github.io/advect/dev/compatibility/scipy/index.md)         | 42 functions and 2 solver adapters behind `advect[scipy]` |
| [xarray](https://yaugenst.github.io/advect/dev/compatibility/xarray/index.md)       | Dynamic labeled-container pytrees behind `advect[xarray]` |

For the raw operation and rule data, use [`advect.support_catalog()`](https://yaugenst.github.io/advect/dev/api/support/index.md).
