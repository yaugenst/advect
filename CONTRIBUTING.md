# Contributing to Advect

Advect stays small by keeping one semantic core and one executable source for
every support claim. For a broad or user-visible change, open a focused issue or
draft pull request before committing to an implementation.

## Set up the repository

Advect requires Python 3.12 or newer and Rust 1.94 or newer:

```bash
uv sync --all-groups
uv lock --check
uv run pre-commit install --install-hooks
rustup toolchain install stable --component clippy rustfmt
```

The hooks are installed per clone. They cover file quality and Conventional
Commit messages; run the full set with `uv run pre-commit run --all-files`.
Avoid `--no-verify`, which bypasses every local check.

The [developer guide](docs/development/index.md) routes code ownership,
operation authoring, tests, and documentation. Read a record from the
[`design/` index](design/README.md) only when the change touches the decision it
owns.

## Submit a change

Work on a feature branch and keep the pull request to one coherent problem. Run
the owning checks while iterating, then every applicable gate in the
[testing guide](docs/development/testing.md). In the pull request, state the
user-visible contract, what you verified, and any unavailable hardware or
remote-CI lane.

Do not add compatibility aliases, duplicate registries, or speculative
fallbacks to make a narrow change look broader. Contributions accepted into
this repository are distributed under the MIT License.

Report suspected vulnerabilities through GitHub's
[private advisory form](https://github.com/yaugenst/advect/security/advisories/new),
not a public issue.
