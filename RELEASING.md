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
- TestPyPI and PyPI trust only `publish-release.yml`, each through its matching
  GitHub environment;
- the `release` environment requires operator approval and accepts deployments
  only from `main`;
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

## Publish

After main CI passes, start the publication workflow from an authorized GitHub
CLI session. The tag names the intended version but must not exist yet:

```bash
gh api --method POST repos/{owner}/{repo}/dispatches \
  -f event_type=release \
  -f 'client_payload[tag]=vX.Y.Z'
```

The default-branch [publication workflow](.github/workflows/publish-release.yml)
pins the current `main` commit, then stops before building unless its
`CI Success` job passed in a `main` push workflow. It builds and smoke-tests the
complete configured native wheel family plus the source distribution, rejects
an incomplete artifact family, checks package metadata, publishes the candidate
to TestPyPI, and verifies a clean scientific install.

After the `Approve production release` job begins waiting, create the annotated
tag at the exact candidate revision shown in the workflow summary, push it, and
approve the `release` environment:

```bash
git tag -a vX.Y.Z CANDIDATE_SHA -m "Advect X.Y.Z"
git push origin vX.Y.Z
```

The workflow verifies that the protected tag still targets the tested candidate
before publishing versioned documentation, creating the GitHub Release,
publishing the same candidate to PyPI, and verifying a clean public install.

TestPyPI files, version tags, and public release files are immutable. If the
TestPyPI candidate is wrong, fix it, choose the next version, and repeat the
release. Wait for the workflow to finish, then confirm that:

- the GitHub Release contains the complete wheel family, one source
  distribution, checksums, and the provenance manifest;
- the PyPI installation smoke passed; and
- `/X.Y.Z/` and `/latest/` serve the released documentation.

The Pyodide wheel belongs to the documentation build, and the Rust crates are
not published separately. Never replace a published artifact or move an
existing version tag; fix the issue under a new version.

## Refresh released documentation

To publish a documentation correction without releasing another package, run
the [Deploy docs workflow](.github/workflows/deploy-docs.yml) manually. Set
`version` to the existing release version `X.Y.Z`, corresponding to the
immutable `vX.Y.Z` tag, and set `source_revision` to the exact commit containing
the documentation to publish.

The deployed version remains the release identity even when its documentation
comes from a later commit. The workflow replaces `/X.Y.Z/`, updates `/latest/`,
and does not change the release tag or package artifacts.
