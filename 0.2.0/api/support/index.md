# Support Catalog

The support catalog supplies the data behind the generated [compatibility pages](https://yaugenst.github.io/advect/0.2.0/compatibility/index.md). It reports which Array API, NumPy, and SciPy calls can run dynamically, be staged, be saved, and be differentiated, together with the capabilities of Advect's internal operations.

## support_catalog

```python
support_catalog() -> dict[str, object]
```

Return live primitive capabilities and supported functions by extension.

Each mode marked true is an end-to-end support claim for the callable's declared frontend contract, rather than a statement that a handler exists.

Examples:

```pycon
>>> import advect as ad
>>> catalog = ad.support_catalog()
>>> catalog["schema_version"]
3
>>> sorted(catalog["extensions"])
['array_api', 'numpy', 'scipy']
```
