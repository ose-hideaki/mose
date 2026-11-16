"""Build the native VBMicrolensing FFI extension from a source checkout.

This is exposed as the console script ``autodiff-vbm-build-native``.  It
requires an editable install of this project (so the ``native`` source tree is
on disk), a C++17 compiler, CMake >= 3.18, and ``VBMicrolensing==5.5``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


def _checkout_root() -> pathlib.Path:
    package_dir = pathlib.Path(__file__).resolve().parent
    root = package_dir.parents[1]
    if not (root / "native" / "CMakeLists.txt").is_file():
        raise RuntimeError(
            "the native source tree is unavailable; install this project as an "
            "editable checkout (`pip install -e '.[vbm]'`) before building"
        )
    return root


def main() -> None:
    root = _checkout_root()
    build = root / "build" / "native"
    output = root / "src" / "autodiff_vbm"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root / "native"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DAUTODIFF_VBM_OUTPUT_DIR={output}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j2"], check=True)


if __name__ == "__main__":
    main()
