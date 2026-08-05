"""Tests for release artifact assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from scripts._support.release_artifacts import ReleaseArtifactError, assemble_release_artifacts

_VERSION = "0.1.0"
_REVISION = "a" * 40
_WHEELS = (
    ("cp312", "cp312", "manylinux_2_17_x86_64.manylinux2014_x86_64"),
    ("cp313", "cp313", "manylinux_2_17_x86_64.manylinux2014_x86_64"),
    ("cp312", "cp312", "macosx_10_12_x86_64"),
    ("cp313", "cp313", "macosx_10_12_x86_64"),
    ("cp312", "cp312", "macosx_11_0_arm64"),
    ("cp313", "cp313", "macosx_11_0_arm64"),
    ("cp312", "cp312", "win_amd64"),
    ("cp313", "cp313", "win_amd64"),
)


def _write_wheel(path: Path) -> None:
    dist_info = f"advect-{_VERSION}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.4\nName: advect\nVersion: {_VERSION}\n",
        )
        archive.writestr(f"{dist_info}/licenses/LICENSE", "MIT\n")


def _write_sdist(path: Path) -> None:
    root = f"advect-{_VERSION}"
    with tarfile.open(path, "w:gz") as archive:
        for relative in (
            "Cargo.toml",
            "LICENSE",
            "pyproject.toml",
            "packages/advect-native/Cargo.toml",
        ):
            payload = b"release fixture\n"
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _write_release_set(dist_dir: Path) -> None:
    dist_dir.mkdir()
    for python_tag, abi_tag, platform_tag in _WHEELS:
        _write_wheel(dist_dir / f"advect-{_VERSION}-{python_tag}-{abi_tag}-{platform_tag}.whl")
    _write_sdist(dist_dir / f"advect-{_VERSION}.tar.gz")


def test_assemble_release_artifacts_validates_and_hashes_complete_set(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    _write_release_set(dist_dir)
    manifest_path = tmp_path / "RELEASE-PROVENANCE.json"
    checksums_path = tmp_path / "SHA256SUMS"

    records = assemble_release_artifacts(
        dist_dir,
        version=_VERSION,
        source_revision=_REVISION,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
    )

    assert len(records) == 9
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_revision"] == _REVISION
    assert manifest["package_version"] == _VERSION
    assert {artifact["filename"] for artifact in manifest["artifacts"]} == {
        path.name for path in dist_dir.iterdir()
    }
    for line in checksums_path.read_text().splitlines():
        digest, filename = line.split("  ")
        assert digest == hashlib.sha256((dist_dir / filename).read_bytes()).hexdigest()


def test_assemble_release_artifacts_rejects_an_incomplete_wheel_family(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    _write_release_set(dist_dir)
    next(dist_dir.glob("*win_amd64.whl")).unlink()

    with pytest.raises(ReleaseArtifactError, match="release wheel family mismatch"):
        assemble_release_artifacts(
            dist_dir,
            version=_VERSION,
            source_revision=_REVISION,
            manifest_path=tmp_path / "RELEASE-PROVENANCE.json",
            checksums_path=tmp_path / "SHA256SUMS",
        )


def test_release_artifact_script_entrypoint_is_directly_runnable() -> None:
    script = Path(__file__).parents[4] / "scripts" / "assemble_release_artifacts.py"
    subprocess.run(  # noqa: S603 - fixed interpreter and repository script
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
