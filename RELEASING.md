# Releasing Advect

[`Cargo.toml`](Cargo.toml) is the only version authority. Python package
metadata and both Rust crates derive their version from it;
[`Cargo.lock`](Cargo.lock) is the generated mirror.

## One-time setup

- Create `testpypi` and `pypi` GitHub environments.
- Register trusted publishers for `yaugenst/advect`, `release.yml`, and the
  matching environment on [TestPyPI](https://test.pypi.org/manage/account/publishing/)
  and [PyPI](https://pypi.org/manage/account/publishing/). No API-token secret
  is needed.
- Configure GitHub Pages to serve the root of the `gh-pages` branch. Before the
  first public release, remove the old deployment with
  `uv run mike delete --push dev`.

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

The tag rebuilds and validates the distributions, deploys the documentation,
creates the GitHub Release, publishes the distributions to PyPI through trusted
publishing, and verifies a clean public install. The GitHub Release contains
the 15 wheels, source distribution, checksums, and provenance manifest. The
Pyodide wheel belongs to the documentation build; the Rust crates are not
published separately.

## Documentation versions

Ordinary CI builds the docs and runs every native and Pyodide snippet, but does
not deploy them. Each release publishes an immutable `/X.Y.Z/` version with
Mike, moves `/latest/` to that release, and makes `latest` the site root. Older
release versions remain available; there is no public `dev` version. Add a
version picker when a second public version makes it useful.
