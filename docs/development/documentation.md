# Documentation

Advect documents the same architecture at different altitudes. Public pages
teach supported behavior; developer pages explain ownership; module docstrings
state local responsibility; scoped `AGENTS.md` files route automated work to
those authorities.

## Published documentation

Put each statement at the narrowest level that owns it:

- `docs/tutorials/` teaches complete user tasks with runnable examples.
- `docs/architecture.md` explains the public execution and extension model.
- `docs/api/` is organized by public import path and renders object docstrings
  with mkdocstrings.
- The NumPy, Array API, CuPy, and SciPy pages under `docs/compatibility/` are
  generated from live support data. The xarray page is a hand-written
  structure contract.
- `docs/development/` is the contributor authority for code ownership,
  operation authoring, testing, and documentation.
- `design/` records requirements, decisions, implementation status, and
  evidence contracts; it is not a second public API reference.

Keep exhaustive callable inventories in the generated compatibility report or
mkdocstrings output. Narrative pages explain behavior and boundaries,
then link to the owning inventory.

## Module and object docstrings

Every production Python module and package, including type stubs and package
initializers, must start with a module docstring. A substantial module should
state:

1. what responsibility it owns;
2. why that responsibility is separate;
3. which neighboring layer it consumes or serves; and
4. the important responsibility it deliberately does not own.

One sentence is enough for a cohesive leaf whose responsibility and boundary
are already clear from its name and types. Documentation checks enforce that
docstrings exist; they deliberately do not turn subjective prose quality into
a word-count gate. Review substantial modules against the four questions
above.

Do not enumerate private leaf files or repeat a directory map in every module.
Every public function and class needs an accurate source docstring. When an
Advect API has non-obvious selection, result structure, errors, or resource and
lifetime behavior, use NumPy-style sections to explain those contracts and add
a short example when it helps. A simple parity wrapper may instead rely on its
accurate signature, a concise source docstring, and the page-level or linked
upstream contract; do not duplicate the upstream manual parameter by
parameter. Private helpers need a docstring when their invariant cannot be
expressed clearly by their name and types.

The public reference uses explicit mkdocstrings directives. Add a public export
to its owning API page; do not place the same directive on several pages.

## Repository guidance

`CONTRIBUTING.md` is the short human doorway. Root `AGENTS.md` routes automated
work and holds only irreducible repository-wide rules. Add a scoped
`AGENTS.md` only where ownership or verification materially changes. Each
scoped file contains four sections: `Owns`, `Must not own`, `Read`, and
`Verify`, with links back to this developer guide or a canonical design source.

Use a subsystem `README.md` for a stable human explanation of a component, as
the two Rust crates do. Do not create a README or agent file for every
directory; redundant guidance drifts and costs readers context.

## Generated compatibility pages

Do not hand-edit the generated index, NumPy, Array API, CuPy, or SciPy pages.
Regenerate them after a public NumPy, Array API, or SciPy support change:

```bash
uv run python -m scripts.report_extension_support \
  --format markdown \
  --output docs/compatibility
```

Tests pin the generated pages to the live declarations.
Update `docs/compatibility/xarray.md` directly when its labeled-container
contract changes; it is intentionally narrative rather than generated.

## Runnable examples and browser docs

Mark a runnable Python fence with ```` ```{.python .run} ````. A page is one
Python session: running a block first executes any unexecuted runnable blocks
above it. Each marked block must therefore run cleanly in order with only
NumPy, Advect, and anything defined by earlier marked blocks. Every runnable
block must also print a meaningful result so its browser output confirms what
the example demonstrated.

Run all snippets natively:

```bash
uv run python scripts/run_doc_snippets.py docs
```

For browser-facing changes, build the wheel once before the strict docs build;
the MkDocs hook stages the wheel, playground adapter, and examples:

```bash
mkdir -p dist
uvx --from pyodide-build==0.36.0 pyodide build . --outdir dist/pyodide
uv run mkdocs build --strict
```

The Pyodide CI lane also runs the snippets inside its browser-compatible
environment.

## Build and LLM index

Run `uv run mkdocs build --strict` for every documentation change. The build
checks navigation, internal links, mkdocstrings imports, snippets included by
Markdown, and the generated `llms.txt`/Markdown bundle. Keep the llms sections
focused on canonical public and developer pages so an automated reader can
find ownership without loading the design archive first.
