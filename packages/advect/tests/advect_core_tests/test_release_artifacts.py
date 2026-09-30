"""Tests for release artifact assembly."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from scripts._support import release_artifacts
from scripts._support.release_artifacts import ReleaseArtifactError, assemble_release_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable

_VERSION = "1.2.3"
_REVISION = "a" * 40
_LICENSE_FILES = tuple(sorted(release_artifacts._REQUIRED_LICENSE_FILES))
_PACKAGE_FILES = ("advect/_native_core.pyi", "advect/py.typed")
_DIST_INFO = f"advect-{_VERSION}.dist-info"
_WHEELS = tuple(
    (python_tag, python_tag, platform_tag)
    for python_tag, platform_tag in product(
        release_artifacts._PYTHON_TAGS,
        (
            "manylinux_2_17_x86_64.manylinux2014_x86_64",
            "manylinux_2_17_aarch64.manylinux2014_aarch64",
            "macosx_10_12_x86_64",
            "macosx_11_0_arm64",
            "win_amd64",
        ),
    )
)
_SDIST_FILES = (
    "Cargo.toml",
    "pyproject.toml",
    "packages/advect-native/Cargo.toml",
    *_LICENSE_FILES,
    *(f"packages/advect/src/{name}" for name in _PACKAGE_FILES),
)


def _metadata(*, version: str = _VERSION, licenses: tuple[str, ...] = _LICENSE_FILES) -> str:
    return f"Metadata-Version: 2.4\nName: advect\nVersion: {version}\n" + "".join(
        f"License-File: {name}\n" for name in licenses
    )


def _write_wheel(path: Path, update: Callable[[dict[str, str]], object] | None = None) -> None:
    entries = {
        f"{_DIST_INFO}/METADATA": _metadata(),
        **{f"{_DIST_INFO}/licenses/{name}": "license fixture\n" for name in _LICENSE_FILES},
        **dict.fromkeys(_PACKAGE_FILES, "package fixture\n"),
    }
    if update is not None:
        update(entries)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _write_sdist(path: Path, *, missing: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative in _SDIST_FILES:
            if relative == missing:
                continue
            payload = b"release fixture\n"
            info = tarfile.TarInfo(f"advect-{_VERSION}/{relative}")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _write_release_set(dist_dir: Path) -> None:
    dist_dir.mkdir()
    for python_tag, abi_tag, platform_tag in _WHEELS:
        _write_wheel(dist_dir / f"advect-{_VERSION}-{python_tag}-{abi_tag}-{platform_tag}.whl")
    _write_sdist(dist_dir / f"advect-{_VERSION}.tar.gz")


def _wheel(dist_dir: Path) -> Path:
    return next(dist_dir.glob("*.whl"))


def _sdist(dist_dir: Path) -> Path:
    return dist_dir / f"advect-{_VERSION}.tar.gz"


def _rejection(
    name: str,
    mutate: Callable[[Path], object],
    match: str,
    revision: str = _REVISION,
) -> object:
    return pytest.param(mutate, match, revision, id=name)


def test_release_command_validates_and_hashes_complete_set(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dist_dir = tmp_path / "dist"
    manifest_path = tmp_path / "RELEASE-PROVENANCE.json"
    checksums_path = tmp_path / "SHA256SUMS"
    _write_release_set(dist_dir)

    returncode = release_artifacts.main(
        [
            *("--dist-dir", str(dist_dir), "--version", _VERSION),
            *("--source-revision", _REVISION, "--manifest-path", str(manifest_path)),
            *("--checksums-path", str(checksums_path)),
        ]
    )

    assert returncode == 0
    assert (
        capsys.readouterr().out == f"validated {len(_WHEELS) + 1} distributions for advect 1.2.3\n"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_revision"] == _REVISION
    assert manifest["package_version"] == _VERSION
    assert {artifact["filename"] for artifact in manifest["artifacts"]} == {
        path.name for path in dist_dir.iterdir()
    }
    for line in checksums_path.read_text().splitlines():
        digest, filename = line.split("  ")
        assert digest == hashlib.sha256((dist_dir / filename).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "match", "revision"),
    [
        _rejection("source-revision", lambda _dist: None, "one full lowercase Git", "A" * 40),
        _rejection("no-sdist", lambda dist: _sdist(dist).unlink(), "exactly one sdist, found 0"),
        _rejection(
            "wheel-family",
            lambda dist: next(dist.glob("*cp315-cp315-win_amd64.whl")).unlink(),
            "release wheel family mismatch",
        ),
        *(
            _rejection(
                name,
                lambda dist, filename=filename: shutil.copy(_wheel(dist), dist / filename),
                match,
            )
            for name, filename, match in (
                ("wheel-name-shape", "advect-1.2.3-cp312-cp312.whl", "unexpected shape"),
                ("distribution", "other-1.2.3-cp312-cp312-win_amd64.whl", "wrong distribution"),
                ("filename-version", "advect-9.9.9-cp312-cp312-win_amd64.whl", "has version 9.9.9"),
                ("platform-tag", "advect-1.2.3-cp312-cp312-linux_x86_64.whl", "platform tag"),
            )
        ),
        _rejection(
            "metadata-count",
            lambda dist: _write_wheel(
                _wheel(dist), lambda entries: entries.pop(f"{_DIST_INFO}/METADATA")
            ),
            "exactly one METADATA file",
        ),
        _rejection(
            "packaged-license",
            lambda dist: _write_wheel(
                _wheel(dist), lambda entries: entries.pop(f"{_DIST_INFO}/licenses/LICENSE")
            ),
            "missing packaged license files: LICENSE$",
        ),
        _rejection(
            "declared-license",
            lambda dist: _write_wheel(
                _wheel(dist),
                lambda entries: entries.update({f"{_DIST_INFO}/METADATA": _metadata(licenses=())}),
            ),
            "metadata is missing License-File entries",
        ),
        *(
            _rejection(
                f"wheel-{name}",
                lambda dist, name=name: _write_wheel(
                    _wheel(dist), lambda entries: entries.pop(name)
                ),
                f"missing required package files: {name}$",
            )
            for name in _PACKAGE_FILES
        ),
        _rejection(
            "unreadable-wheel",
            lambda dist: _wheel(dist).write_bytes(b"not a zip"),
            "not a readable wheel",
        ),
        _rejection(
            "metadata-version",
            lambda dist: _write_wheel(
                _wheel(dist),
                lambda entries: entries.update({f"{_DIST_INFO}/METADATA": _metadata(version="0")}),
            ),
            "metadata does not identify advect 1.2.3",
        ),
        _rejection(
            "sdist-name",
            lambda dist: _sdist(dist).rename(dist / "advect-9.9.9.tar.gz"),
            "source distribution is advect-9.9.9.tar.gz",
        ),
        _rejection(
            "unreadable-sdist",
            lambda dist: _sdist(dist).write_bytes(b"not a tar"),
            "not a readable source distribution",
        ),
        *(
            _rejection(
                f"sdist-{name}",
                lambda dist, name=name: _write_sdist(_sdist(dist), missing=name),
                f"missing required source files: advect-{_VERSION}/{name}$",
            )
            for name in _SDIST_FILES
        ),
    ],
)
def test_assemble_release_artifacts_rejects_an_invalid_set_without_provenance(
    tmp_path: Path,
    mutate: Callable[[Path], object],
    match: str,
    revision: str,
) -> None:
    _write_release_set(tmp_path / "dist")
    mutate(tmp_path / "dist")

    with pytest.raises(ReleaseArtifactError, match=match):
        assemble_release_artifacts(
            tmp_path / "dist",
            version=_VERSION,
            source_revision=revision,
            manifest_path=tmp_path / "provenance" / "RELEASE-PROVENANCE.json",
            checksums_path=tmp_path / "provenance" / "SHA256SUMS",
        )

    assert not (tmp_path / "provenance").exists()


def test_release_artifact_script_entrypoint_is_directly_runnable() -> None:
    script = Path(__file__).parents[4] / "scripts" / "assemble_release_artifacts.py"
    subprocess.run(  # noqa: S603 - fixed interpreter and repository script
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
