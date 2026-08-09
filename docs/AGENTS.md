## Owns

- Published tutorials, architecture explanation, API reference, compatibility
  documentation, playground documentation, and contributor guidance.
- Mkdocstrings placement, runnable examples, navigation, and the LLM document
  index.

## Must not own

- A second operation inventory, support registry, design decision archive, or
  implementation-status record.
- Hand edits to the generated compatibility index, NumPy, Array API, CuPy, or
  SciPy pages; xarray is the deliberate narrative exception.
- Duplicate mkdocstrings directives or scoped guidance in every directory.

## Read

- [Documentation system](development/documentation.md)
- [Developer guide](development/index.md)
- [Public architecture](architecture.md)
- The source module docstrings for any API page being changed

## Verify

- `uv run mkdocs build --strict`
- `uv run python scripts/run_doc_snippets.py docs`
- Inspect the rendered page for a visual or interactive change.
- Build and test the Pyodide wheel when runnable browser behavior changes, as
  described in the [documentation guide](development/documentation.md).
