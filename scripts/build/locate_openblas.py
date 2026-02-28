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

from pathlib import Path


def main():
    # Import intentionally local to keep script fast-failing with a clear error.
    import scipy_openblas32 as openblas  # pylint: disable=import-outside-toplevel

    lib_dir = Path(openblas.get_lib_dir()).resolve()
    lib_file = lib_dir / openblas.get_library(fullname=True)
    cmake_dir = (lib_dir / "cmake" / "openblas").resolve()

    if not lib_file.exists():
        raise FileNotFoundError(lib_file)
    if not cmake_dir.exists():
        raise FileNotFoundError(cmake_dir)

    print(f"MQ_OPENBLAS_ROOT={lib_dir.parent.as_posix()}")
    print(f"MQ_OPENBLAS_LIB_DIR={lib_dir.as_posix()}")
    print(f"MQ_OPENBLAS_LIB={lib_file.as_posix()}")
    print(f"MQ_OPENBLAS_CMAKE_DIR={cmake_dir.as_posix()}")
    print("MQ_OPENBLAS_SYMBOL_PREFIX=scipy_")


if __name__ == "__main__":
    main()
