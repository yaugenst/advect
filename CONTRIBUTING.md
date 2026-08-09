# Contributing to Advect

Advect is deliberately small: preserve one semantic core, one canonical
operation registry, and one source of truth for every public support claim. A
focused issue or draft pull request is the best place to align on a broad or
user-visible extension before implementing it.

## Start here

Advect requires Python 3.12 or newer and Rust 1.94 or newer:

```bash
uv sync --all-groups
uv lock --check
uv run pre-commit install --install-hooks
rustup toolchain install stable --component clippy rustfmt
```

Hook installation is per clone. It installs the file-quality checks used by CI
and the Conventional Commit message check. Run the complete hook set after
setup with `uv run pre-commit run --all-files`. For a deliberate direct commit
to `main`, bypass only the branch guard with
`SKIP=no-commit-to-branch git commit`; do not disable every hook with
`--no-verify`.

The [developer guide](docs/development/index.md) is the contributor authority:

- [codebase map](docs/development/codebase.md) — responsibility and dependency boundaries;
- [adding operations](docs/development/adding-operations.md) — custom and built-in
  primitives, NumPy and SciPy forms, providers, the Python staged-program
  envelope, the Rust graph artifact, and the native adapter and dynamic tape;
- [testing](docs/development/testing.md) — suite ownership and complete local gates;
- [documentation](docs/development/documentation.md) — public pages, docstrings,
  generated compatibility reports, and runnable examples.

Read the relevant design decision only when a change touches its contract. The
[`design/` index](design/README.md) routes requirements, decisions,
implementation status, and performance evidence.

## Submit a change

Work on a feature branch and keep the pull request to one coherent problem.
Run the focused checks while iterating, then every applicable gate in the
[testing guide](docs/development/testing.md). In the pull request, state the
user-visible contract, what was verified, and any unavailable hardware or
remote-CI lane.

Do not add compatibility aliases, duplicated registries, or speculative
fallbacks to make a narrow change appear broader. Contributions accepted into
this repository are distributed under the MIT license.

Report suspected vulnerabilities through the repository's
[private security-advisory form](https://github.com/yaugenst/advect/security/advisories/new),
not a public issue.
