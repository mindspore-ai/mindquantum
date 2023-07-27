#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Console entry point to access MindQuantum installation variables."""

import argparse
import sys
from distutils.util import get_platform  # pylint: disable=deprecated-module
from pathlib import Path

try:
    import importlib.metadata as importlib_metadata  # pragma: no cover (PY38+)
except ImportError:
    import importlib_metadata  # pragma: no cover (<PY38)

_ROOT = Path(__file__).parent.parent.resolve()

# ==============================================================================


def print_includes():
    """Print a list of include directories using the -I<dir> syntax."""
    dirs = [
        _ROOT / 'mindquantum' / 'include' / 'mindquantum',
        _ROOT / 'ccsrc' / 'include',
        _ROOT / 'ccsrc' / 'python' / 'core' / 'include',
        _ROOT / 'ccsrc' / 'python' / 'mqbackend' / 'include',
        _ROOT / 'ccsrc' / 'python' / 'simulator' / 'include',
    ]

    root = _ROOT / 'mindquantum' / 'lib' / 'mindquantum' / 'third_party'
    if root.exists():
        for folder in root.iterdir():
            if (folder / 'include').exists():
                dirs.append(folder / 'include')
            else:
                dirs.append(folder)

    root = _ROOT / 'build' / '.mqlibs'
    if root.exists():
        for folder in root.iterdir():
            dirs.append(folder / 'include')

    unique_dirs = []
    for folder in dirs:
        if folder and folder.exists() and folder not in unique_dirs:
            unique_dirs.append(folder)

    print(' '.join(f'-I{d}' for d in unique_dirs))


# ==============================================================================


def get_cmake_dir(as_string=True):
    """
    Return the path to the MindQuantum CMake module directory.

    Args:
        as_string (bool): (optional) If true, returned value is a string, else a pathlib.Path object.
    """

    def get_dir(folder):
        """Convert a pathlib.Path to a string if requested."""
        if as_string:
            return str(folder)
        return folder

    cmake_installed_path = Path(_ROOT, 'mindquantum', 'share', 'mindquantum', 'cmake')
    if cmake_installed_path.exists():
        return get_dir(cmake_installed_path)

    build_dir = Path(_ROOT, 'build')
    if build_dir.exists():
        return get_dir(build_dir)

    raise ImportError('MindQuantum not installed, installation required to access the CMake files')


# ==============================================================================


def print_bin_dir():
    """Print the bin directory of MindQuantum."""
    bin_path = Path(_ROOT, "mindquantum", "bin")
    if bin_path.exists():
        print(str(bin_path))
        return
    raise ImportError("MindQuantum not installed.")


# ==============================================================================


def print_abi():
    """Print the bin directory of MindQuantum."""
    try:
        from mindquantum import mqbackend  # pylint: disable=import-outside-toplevel

        print(mqbackend.c.build_abi())  # pylint: disable=no-member
    except ImportError as exc:
        raise ImportError("MindQuantum not installed.") from exc


# ==============================================================================


def print_temp_dir():
    """Print the default build directory used by setup.py."""
    # Based on setuptools/_distutils/command/build.py
    print(str(Path('build') / f"temp.{get_platform()}-{sys.implementation.cache_tag}" / _ROOT.name))


# ==============================================================================


def print_version():
    """Print MindQuantum's version."""
    try:
        print(importlib_metadata.version('mindquantum'))
    except importlib_metadata.PackageNotFoundError:
        with (_ROOT / 'VERSION.txt').open() as fd:
            print([line.strip() for line in fd.readlines() if line][0])


# ==============================================================================


def main():
    """Implement main functionality."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cmakedir",
        action="store_true",
        help=(
            "Print the CMake module directory, ideal for setting either -Dmindquantum_ROOT or-Dmindquantum_DIR in "
            "CMake."
        ),
    )
    group.add_argument(
        "--includes",
        action="store_true",
        help="Include flags for MindQuantum",
    )
    group.add_argument(
        "--tempdir",
        action="store_true",
        help="Print the default build directory used by setup.py",
    )
    group.add_argument(
        "--version",
        action="store_true",
        help="Print out MindQuantum's version",
    )
    group.add_argument(
        "--bin",
        action="store_true",
        help="Print bin directory of MindQuantum",
    )
    group.add_argument(
        "--abi",
        action="store_true",
        help="Print abi of MindQuantum backend",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.version:
        print_version()
    if args.includes:
        print_includes()
    if args.cmakedir:
        print(get_cmake_dir())
    if args.bin:
        print_bin_dir()
    if args.tempdir:
        print_temp_dir()
    if args.abi:
        print_abi()


if __name__ == "__main__":
    main()
