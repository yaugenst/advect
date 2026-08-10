# Releasing Advect

[`Cargo.toml`](Cargo.toml) is the only version authority. Python package
metadata and both Rust crates derive their version from it;
[`Cargo.lock`](Cargo.lock) is the generated mirror.

## Check release state

Before changing the version, confirm that:

- `main` is current and its required CI checks pass;
- the target version is unused as a Git tag and on TestPyPI and PyPI;
- the Release workflow's TestPyPI and PyPI publishing identities are active;
- versioned documentation can deploy to `gh-pages`; and
- you can push an annotated release tag and inspect the resulting workflow.

## Prepare a version

1. Change `workspace.package.version` in `Cargo.toml`.
2. Run `cargo check --workspace` to update `Cargo.lock`.
3. Run `uv lock --check` and the applicable checks in the
   [testing guide](docs/development/testing.md).
4. Commit the two version files as `chore(release): prepare X.Y.Z`, land that
   commit on `main`, and wait for CI.

Do not use `cz bump`; Commitizen validates commit messages but does not own the
package version.

## Rehearse on TestPyPI

Run the [Release workflow](.github/workflows/release.yml) manually on `main`.
It builds and smoke-tests 15 native wheels plus the source distribution,
rejects an incomplete artifact family, checks package metadata, publishes the
whole candidate to TestPyPI, and verifies a clean scientific install from
TestPyPI.

TestPyPI files are treated as immutable. If the candidate is wrong, fix it,
choose the next version, and repeat the rehearsal.

## Publish

After the TestPyPI run and main CI both pass, tag that exact commit:

```bash
git tag -a vX.Y.Z -m "Advect X.Y.Z"
git push origin vX.Y.Z
```

The tag rebuilds and validates the distributions, publishes versioned
documentation, creates the GitHub Release, publishes to PyPI, and verifies a
clean public install. Wait for the workflow to finish, then confirm that:

- the GitHub Release contains 15 wheels, one source distribution, checksums,
  and the provenance manifest;
- the PyPI installation smoke passed; and
- `/X.Y.Z/` and `/latest/` serve the released documentation.

The Pyodide wheel belongs to the documentation build, and the Rust crates are
not published separately. Never replace a published artifact or move an
existing version tag; fix the issue under a new version.
