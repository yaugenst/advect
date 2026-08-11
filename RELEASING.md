# Releasing Advect

[`Cargo.toml`](Cargo.toml) is the only version authority. Python package
metadata and both Rust crates derive their version from it;
[`Cargo.lock`](Cargo.lock) is the generated mirror.

The binary release family targets CPython 3.12 through 3.14 on manylinux
x86-64 and arm64, macOS Intel and Apple silicon, and Windows x86-64. Release
one does not target musllinux, Windows ARM64, free-threaded CPython, PyPy, or
abi3 wheels.

## Check release state

Before changing the version, confirm that:

- `main` is current and its required CI checks pass;
- the target version is unused as a Git tag and on TestPyPI and PyPI;
- TestPyPI trusts only `rehearse-release.yml`, and PyPI trusts only
  `publish-release.yml`;
- the retired legacy `release.yml` workflow is disabled in GitHub Actions;
- the publishing environments allow only the `main`-anchored release workflows
  and exclude tag refs;
- versioned documentation can deploy to `gh-pages`; and
- the release operator can send a repository dispatch with contents-write access.

## Prepare a version

1. Change `workspace.package.version` in `Cargo.toml`.
2. Run `cargo check --workspace` to update `Cargo.lock`.
3. Run `uv sync --all-groups --locked` and the applicable checks in the
   [testing guide](docs/development/testing.md).
4. Commit the two version files as `chore(release): prepare X.Y.Z`, land that
   commit on `main`, and wait for CI.

Do not use `cz bump`; Commitizen validates commit messages but does not own the
package version.

## Rehearse on TestPyPI

Run the [release rehearsal](.github/workflows/rehearse-release.yml) manually on
`main` with the planned `vX.Y.Z` tag. It builds and smoke-tests the complete
configured native wheel family plus the source distribution, rejects an
incomplete artifact family, checks package metadata, publishes the whole
candidate to TestPyPI, and verifies a clean scientific install from TestPyPI.

TestPyPI files are treated as immutable. If the candidate is wrong, fix it,
choose the next version, and repeat the rehearsal.

## Publish

After the TestPyPI run and main CI both pass, tag that exact commit:

```bash
git tag -a vX.Y.Z -m "Advect X.Y.Z"
git push origin vX.Y.Z
```

The supported publication path is a separate repository dispatch from an
authorized GitHub CLI session:

```bash
gh api --method POST repos/{owner}/{repo}/dispatches \
  -f event_type=release \
  -f 'client_payload[tag]=vX.Y.Z'
```

The default-branch [publication workflow](.github/workflows/publish-release.yml)
resolves the annotated tag to an immutable commit, then stops before building
unless that exact commit is on `main` and its `CI Success` job passed in a
`main` push workflow. It rebuilds and validates the distributions, publishes
versioned documentation, creates the GitHub Release, publishes to PyPI, and
verifies a clean public install. Wait for the workflow to finish, then confirm
that:

- the GitHub Release contains the complete wheel family, one source
  distribution, checksums, and the provenance manifest;
- the PyPI installation smoke passed; and
- `/X.Y.Z/` and `/latest/` serve the released documentation.

The Pyodide wheel belongs to the documentation build, and the Rust crates are
not published separately. Never replace a published artifact or move an
existing version tag; fix the issue under a new version.
