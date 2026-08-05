from __future__ import annotations

import subprocess
import sys

import advect as ad
from advect import autodiff
from advect._autodiff_exports import AUTODIFF_EXPORT_MODULES
from advect.autodiff import api as autodiff_api


def test_import_and_version() -> None:
    # Package should import from source layout and expose a version string
    assert isinstance(ad.__version__, str)
    assert ad.__version__


def test_public_autodiff_facades_share_one_export_inventory() -> None:
    names = list(AUTODIFF_EXPORT_MODULES)
    assert autodiff.__all__ == names
    assert autodiff_api.__all__ == names
    assert set(names) <= set(ad.__all__)
    for name in names:
        assert getattr(ad, name) is getattr(autodiff, name)
        assert getattr(autodiff, name) is getattr(autodiff_api, name)


def test_root_import_keeps_autodiff_lazy() -> None:
    subprocess.run(  # noqa: S603 - fixed interpreter and source string
        [
            sys.executable,
            "-c",
            "import sys; import advect; "
            "assert not any(name == 'advect.autodiff' or "
            "name.startswith('advect.autodiff.') for name in sys.modules)",
        ],
        check=True,
    )
