# AGENTS

Keep Advect small. Prefer one source of truth, deletion, and narrow contracts
over compatibility layers or speculative fallbacks. Run the direct checks that
own the behavior you change.

## Route the task

Read the nearest scoped `AGENTS.md`, then the relevant developer page:

- [Developer guide](docs/development/index.md)
- [Codebase map and dependency boundaries](docs/development/codebase.md)
- [Adding operations and public forms](docs/development/adding-operations.md)
- [Test ownership and commands](docs/development/testing.md)
- [Documentation system](docs/development/documentation.md)

Read a design document or ADR only when the change touches the decision it
owns. [`design/README.md`](design/README.md) is the index.

For an ownership or workflow question, stop when the nearest scoped guidance
and the relevant developer page answer it. Read source, CI, or design records
only to implement a change, confirm current wiring or completeness, diagnose
observed behavior, or fill a concrete gap in the guides.

## Global rules

- Python source lives in `packages/advect/src`; use absolute imports.
- Keep `advect.core` standard-library-only. Third-party dependencies belong in
  frontends and explicit optional adapters.
- NumPy and Array API frontends meet at canonical operation emission. Do not
  create a second operation registry or support inventory.
- Prefer JVP-first derivative rules. Never silently substitute numerical
  pullbacks.
- Python core owns the versioned `StagedProgram` envelope and validates its
  cross-record consistency. `advect-runtime` owns the enclosed canonical graph
  artifact and `GraphStore`; Python owns no parallel graph model.
- `advect-native` owns Python/Rust translation and the invocation-local dynamic
  tape, not either durable format.
- Do not add compatibility aliases, deprecated dual paths, duplicated metadata,
  or dead code.
- Preserve domain invariants, ordinary error reporting, destructive-operation
  safeguards, and process or resource cleanup.

## Workflow

- Work on a feature branch. Rebase on current `origin/main` before merge and
  keep history linear; merge pull requests with rebase.
- Use Python 3.12 or newer and Rust 1.94 or newer. Set up with
  `uv sync --all-groups`, `uv lock --check`, and
  `rustup toolchain install stable --component clippy rustfmt` as needed.
- While iterating, run the focused suite named by the nearest scoped guidance.
  Before review, run every applicable gate in the [testing guide](docs/development/testing.md).
- When finishing, state what was verified, what was not, and any remaining
  hardware or remote-CI gap. If Git or GitHub state changed, report branch,
  push, pull-request, and comment state separately.
