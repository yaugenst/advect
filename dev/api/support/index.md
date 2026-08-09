# Support Catalog

The support catalog is the machine-readable authority behind the generated compatibility pages. It reports canonical primitive capabilities and public Array API, NumPy, and SciPy forms. Dynamic, staged, and serialized are separate lifetime claims; registration alone does not create a public claim.

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
