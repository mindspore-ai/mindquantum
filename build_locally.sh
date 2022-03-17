#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================
# Default values

do_clean=0
do_clean_build_dir=0
do_clean_cache=0
do_clean_venv=0
enable_gpu=0
n_jobs=$(nproc)

source_dir=$(realpath "$BASEPATH")
build_dir=$(realpath "$source_dir/build")

# ==============================================================================

call_cmake() {
    echo "**********"
    echo "Calling CMake with: cmake " "$@"
    echo "**********"
    cmake "$@"
}

# ==============================================================================

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
no_arg() {
    if [ -n "$OPTARG" ]; then die "No arg allowed for --$OPT option"; fi; }
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# ------------------------------------------------------------------------------

help_message() {
    echo 'Build MindQunantum locally (in-source build)'
    echo ''
    echo 'This is mainly relevant for developers that do not want to always '
    echo 'have to reinstall the Python package'
    echo ''
    echo 'This script will create a Python virtualenv in the MindQuantum root'
    echo 'directory and then build all the C++ Python modules and place the'
    echo 'generated libraries in their right locations within the MindQuantum'
    echo 'folder hierarchy so Python knows how to find them.'
    echo ''
    echo 'A pth-file will be created in the virtualenv site-packages directory'
    echo 'so that the MindQuantum root folder will be added to the Python PATH'
    echo 'without the need to modify PYTHONPATH.'
    echo -e '\nUsage:'
    echo "  $(basename "$0") [options] [-- cmake_options]"
    echo -e '\nOptions:'
    echo '  -h,--help          Show this help message and exit'
    echo '  -B [dir]           Specify build directory'
    echo "                     Defaults to: $build_dir"
    echo '  -c,--clean         Run make clean before building'
    echo '  --clean-all        Clean everything before building.'
    echo '                     Equivalent to --clean-venv --clean-builddir'
    echo '  --clean-builddir   Delete build directory before building'
    echo '  --clean-cache      Re-run CMake to generate the CMake cache'
    echo '  --clean-venv       Clean Python virtualenv before building'
    echo '  --gpu              Enable GPU support'
    echo '  -j,--jobs [N]      Number of parallel jobs for building'
    echo "                     Defaults to: $n_jobs"
    echo ''
    echo 'Any options after "--" will be passed onto CMake when configuring'
    echo -e '\nExample calls:'
    echo "$0 -B build"
    echo "$0 -B build --gpu"
    echo "$0 -B build -- -DIN_PLACE_BUILD=ON"
}

# ==============================================================================

while getopts hcB:j:-: OPT; do
    if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi
    # shellcheck disable=SC2214
    case "$OPT" in
        h | help )       no_arg;
                         help_message >&2
                         exit 1 ;;
        B )              needs_arg;
                         build_dir="$OPTARG"
                         ;;
        c | clean )      no_arg;
                         do_clean=1
                         ;;
        clean-all )      no_arg;
                         do_clean_venv=1
                         do_clean_build_dir=1
                         ;;
        clean-builddir ) no_arg;
                         do_clean_build_dir=1
                         ;;
        clean-cache )    no_arg;
                         do_clean_cache=1
                         ;;
        clean-venv )     no_arg;
                         do_clean_venv=1
                         ;;
        gpu )            no_arg;
                         enable_gpu=1
                         ;;
        j | jobs )       needs_arg;
                         n_jobs="$OPTARG"
                         ;;
        ??* )           die "Illegal option --OPT: $OPT" ;;
        \? )            exit 2 ;;  # bad short option (error reported via getopts)
    esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# ==============================================================================

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python3
else
    echo 'Unable to locate python or python3!' 1>&2
    exit 1
fi

# ==============================================================================

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}


# ==============================================================================

set -e

cd "${BASEPATH}"

# ------------------------------------------------------------------------------
# Create a virtual environment for building the wheel

if [ $do_clean_venv -eq 1 ]; then
    echo "Deleting virtualenv folder: $BASEPATH/venv"
    rm -rf venv
fi

if [ $do_clean_build_dir -eq 1 ]; then
    echo "Deleting build folder: $build_dir"
    rm -rf "$build_dir"
fi

created_venv=0
if [ ! -d "$BASEPATH/venv" ]; then
    created_venv=1
    echo "Creating Python virtualenv: $PYTHON -m venv venv"
    $PYTHON -m venv venv
fi

source venv/bin/activate

if [ $created_venv -eq 1 ]; then
    pkgs=(pip setuptools wheel build pybind11)

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pkgs+=(auditwheel)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        pkgs+=(delocate)
    fi

    echo "Updating Python packages: $PYTHON -m pip install -U ${pkgs[*]}"
    $PYTHON -m pip install -U "${pkgs[@]}"
fi

# Make sure the root directory is in the virtualenv PATH
site_pkg_dir=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
pth_file="$site_pkg_dir/mindquantum_local.pth"

if [ ! -e "$pth_file" ]; then
    echo "Creating pth-file in $pth_file"
    echo "$BASEPATH" > "$pth_file"
fi

# ------------------------------------------------------------------------------
# Setup arguments for build

cmake_args=(-DENABLE_PROJECTQ:BOOL=ON -DENABLE_QUEST:BOOL=OFF -DIN_PLACE_BUILD:BOOL=ON)

if [[ $enable_gpu -eq 1 ]]; then
    cmake_args+=(-DENABLE_CUDA:BOOL=ON)
fi

# ------------------------------------------------------------------------------
# Build

do_configure=0
if [ ! -d "$build_dir" ]; then
    do_configure=1
elif [ $do_clean_cache -eq 1 ]; then
    do_configure=1
    echo "Removing CMake cache at: $build_dir/CMakeCache.txt"
    rm -f "$build_dir/CMakeCache.txt"
fi

if [ $do_configure -eq 1 ]; then
    call_cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}" "$@"
fi

if [ $do_clean -eq 1 ]; then
    call_cmake --build "$build_dir" --target clean
fi

call_cmake --build "$build_dir" --target all -j "$n_jobs"

# ==============================================================================
