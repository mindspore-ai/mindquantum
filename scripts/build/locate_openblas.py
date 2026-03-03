#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Locate OpenBLAS paths from scipy-openblas32."""

from importlib import util as importlib_util
from pathlib import Path
import sys


def _pkg_root():
    """Return the scipy_openblas32 package directory without importing it."""
    spec = importlib_util.find_spec("scipy_openblas32")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("scipy_openblas32")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _pick_lib_file(lib_dir):
    """Pick the best OpenBLAS link target for the current platform."""
    if sys.platform == "win32":
        preferred = sorted(lib_dir.glob("*.lib"))
        fallback = sorted(lib_dir.glob("*.dll"))
    elif sys.platform == "darwin":
        preferred = sorted(lib_dir.glob("libscipy_openblas*.dylib"))
        fallback = sorted(lib_dir.glob("libscipy_openblas*"))
    else:
        preferred = sorted(lib_dir.glob("libscipy_openblas*.so*"))
        fallback = sorted(lib_dir.glob("libscipy_openblas*"))

    if preferred:
        return preferred[0].resolve()
    if fallback:
        return fallback[0].resolve()
    raise FileNotFoundError(f"No OpenBLAS library found under: {lib_dir}")


def main():
    pkg_root = _pkg_root()
    lib_dir = (pkg_root / "lib").resolve()

    if not lib_dir.exists():
        raise FileNotFoundError(lib_dir)

    cmake_dir = (lib_dir / "cmake" / "openblas").resolve()
    lib_file = _pick_lib_file(lib_dir)

    if not cmake_dir.exists():
        raise FileNotFoundError(cmake_dir)

    print(f"MQ_OPENBLAS_ROOT={lib_dir.parent}")
    print(f"MQ_OPENBLAS_LIB_DIR={lib_dir}")
    print(f"MQ_OPENBLAS_LIB={lib_file}")
    print(f"MQ_OPENBLAS_CMAKE_DIR={cmake_dir}")
    print("MQ_OPENBLAS_SYMBOL_PREFIX=scipy_")


if __name__ == "__main__":
    main()
