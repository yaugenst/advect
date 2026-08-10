"""Validate and describe the complete Advect release artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from pathlib import Path

_EXPECTED_WHEELS = frozenset(
    {
        ("cp312", "cp312", "linux-x86_64"),
        ("cp313", "cp313", "linux-x86_64"),
        ("cp314", "cp314", "linux-x86_64"),
        ("cp312", "cp312", "linux-aarch64"),
        ("cp313", "cp313", "linux-aarch64"),
        ("cp314", "cp314", "linux-aarch64"),
        ("cp312", "cp312", "macos-x86_64"),
        ("cp313", "cp313", "macos-x86_64"),
        ("cp314", "cp314", "macos-x86_64"),
        ("cp312", "cp312", "macos-arm64"),
        ("cp313", "cp313", "macos-arm64"),
        ("cp314", "cp314", "macos-arm64"),
        ("cp312", "cp312", "windows-x86_64"),
        ("cp313", "cp313", "windows-x86_64"),
        ("cp314", "cp314", "windows-x86_64"),
    }
)
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
_WHEEL_NAME_PARTS = 5
_REQUIRED_LICENSE_FILES = frozenset(
    {
        "LICENSE",
        "RUST_STDLIB_COPYRIGHT.html",
        "RUST_STDLIB_LICENSE_MIT.txt",
        "RUST_STDLIB_LICENSE_UNICODE_3_0.txt",
        "THIRD_PARTY_LICENSES.txt",
    }
)


class ReleaseArtifactError(ValueError):
    """Raised when a candidate distribution set is incomplete or inconsistent."""


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable provenance for one candidate distribution."""

    filename: str
    kind: str
    sha256: str
    size: int
    python_tag: str | None = None
    abi_tag: str | None = None
    platform_family: str | None = None


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _platform_family(platform_tag: str) -> str:
    tags = set(platform_tag.split("."))
    if tags & {"manylinux_2_17_x86_64", "manylinux2014_x86_64"}:
        return "linux-x86_64"
    if tags & {"manylinux_2_17_aarch64", "manylinux2014_aarch64"}:
        return "linux-aarch64"
    if platform_tag.startswith("macosx_") and platform_tag.endswith("_x86_64"):
        return "macos-x86_64"
    if platform_tag.startswith("macosx_") and platform_tag.endswith("_arm64"):
        return "macos-arm64"
    if platform_tag == "win_amd64":
        return "windows-x86_64"
    message = f"unsupported release wheel platform tag: {platform_tag}"
    raise ReleaseArtifactError(message)


def _parse_wheel_name(path: Path) -> tuple[str, str, str, str]:
    parts = path.name.removesuffix(".whl").split("-")
    if len(parts) != _WHEEL_NAME_PARTS:
        message = f"release wheel name has an unexpected shape: {path.name}"
        raise ReleaseArtifactError(message)
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if distribution != "advect":
        message = f"release wheel has the wrong distribution name: {path.name}"
        raise ReleaseArtifactError(message)
    return version, python_tag, abi_tag, platform_tag


def _validate_wheel(path: Path, *, version: str) -> ArtifactRecord:
    filename_version, python_tag, abi_tag, platform_tag = _parse_wheel_name(path)
    if filename_version != version:
        message = f"{path.name} has version {filename_version}, expected {version}"
        raise ReleaseArtifactError(message)

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                message = f"{path.name} must contain exactly one METADATA file"
                raise ReleaseArtifactError(message)
            metadata_name = metadata_names[0]
            license_root = metadata_name.removesuffix("METADATA") + "licenses/"
            packaged_licenses = {
                name.removeprefix(license_root) for name in names if name.startswith(license_root)
            }
            if missing := sorted(_REQUIRED_LICENSE_FILES - packaged_licenses):
                message = f"{path.name} is missing packaged license files: {', '.join(missing)}"
                raise ReleaseArtifactError(message)
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            declared_licenses = set(metadata.get_all("License-File") or ())
            if missing := sorted(_REQUIRED_LICENSE_FILES - declared_licenses):
                message = (
                    f"{path.name} metadata is missing License-File entries: {', '.join(missing)}"
                )
                raise ReleaseArtifactError(message)
    except zipfile.BadZipFile as error:
        message = f"{path.name} is not a readable wheel"
        raise ReleaseArtifactError(message) from error

    if metadata["Name"] != "advect" or metadata["Version"] != version:
        message = f"{path.name} metadata does not identify advect {version}"
        raise ReleaseArtifactError(message)

    return ArtifactRecord(
        filename=path.name,
        kind="wheel",
        sha256=_sha256(path),
        size=path.stat().st_size,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_family=_platform_family(platform_tag),
    )


def _validate_sdist(path: Path, *, version: str) -> ArtifactRecord:
    expected_name = f"advect-{version}.tar.gz"
    if path.name != expected_name:
        message = f"source distribution is {path.name}, expected {expected_name}"
        raise ReleaseArtifactError(message)

    root = expected_name.removesuffix(".tar.gz")
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
    except tarfile.TarError as error:
        message = f"{path.name} is not a readable source distribution"
        raise ReleaseArtifactError(message) from error

    required = {
        f"{root}/Cargo.toml",
        f"{root}/pyproject.toml",
        f"{root}/packages/advect-native/Cargo.toml",
        *(f"{root}/{name}" for name in _REQUIRED_LICENSE_FILES),
    }
    if missing := sorted(required - names):
        message = f"{path.name} is missing required source files: {', '.join(missing)}"
        raise ReleaseArtifactError(message)

    return ArtifactRecord(
        filename=path.name,
        kind="sdist",
        sha256=_sha256(path),
        size=path.stat().st_size,
    )


def assemble_release_artifacts(
    dist_dir: Path,
    *,
    version: str,
    source_revision: str,
    manifest_path: Path,
    checksums_path: Path,
) -> list[ArtifactRecord]:
    """Validate a complete release set and write its immutable provenance."""
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        message = "source revision must be one full lowercase Git commit SHA"
        raise ReleaseArtifactError(message)

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(sdists) != 1:
        message = f"release set must contain exactly one sdist, found {len(sdists)}"
        raise ReleaseArtifactError(message)

    records = [_validate_wheel(path, version=version) for path in wheels]
    actual_wheels = {
        (record.python_tag, record.abi_tag, record.platform_family) for record in records
    }
    if actual_wheels != _EXPECTED_WHEELS:
        missing = sorted(_EXPECTED_WHEELS - actual_wheels)
        unexpected = sorted(actual_wheels - _EXPECTED_WHEELS)
        message = f"release wheel family mismatch; missing={missing}, unexpected={unexpected}"
        raise ReleaseArtifactError(message)

    records.append(_validate_sdist(sdists[0], version=version))
    records.sort(key=lambda record: record.filename)

    manifest = {
        "schema_version": 1,
        "package": "advect",
        "package_version": version,
        "source_revision": source_revision,
        "artifacts": [asdict(record) for record in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text(
        "".join(f"{record.sha256}  {record.filename}\n" for record in records),
        encoding="utf-8",
    )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the complete Advect release set and write provenance files."
    )
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checksums-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run release artifact assembly from command-line arguments."""
    args = _parser().parse_args(argv)
    records = assemble_release_artifacts(
        args.dist_dir,
        version=args.version,
        source_revision=args.source_revision,
        manifest_path=args.manifest_path,
        checksums_path=args.checksums_path,
    )
    print(f"validated {len(records)} distributions for advect {args.version}")
    return 0
