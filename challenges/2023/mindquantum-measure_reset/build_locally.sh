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

# shellcheck disable=SC2154

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
ROOTDIR="$BASEPATH"
PROGRAM=$(basename "${BASH_SOURCE[0]:-$0}")

# Test for MindSpore CI
_IS_MINDSPORE_CI=0
if [[ "${JENKINS_URL:-0}" =~ https?://build.mindspore.cn && ! "${CI:-0}" =~ ^(false|0)$ ]]; then
    echo "Detected MindSpore/MindQuantum CI"
    # shellcheck disable=SC2034
    _IS_MINDSPORE_CI=1
fi

echo "Called with: $*"

# ==============================================================================

# Load common bash helper functions
. "$ROOTDIR/scripts/build/common_functions.sh"

# -----------------------------------------------------------------------------

function help_header() {
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
}

function extra_help() {
    echo 'Extra options:'
    echo '  --clean              Run make clean before building'
    echo '  -c,--configure       Force running the CMake configure step'
    echo '  --configure-only     Stop after the CMake configure and generation steps (ie. before building MindQuantum)'
    echo '  --doc,--docs         Setup the Python virtualenv for building the documentation and ask CMake to build the'
    echo '                       documentation'
    echo '  --install            Build the ´install´ target'
    echo '  --prefix             Specify installation prefix'
    echo ''
    echo 'Any options after "--" will be passed onto CMake during the configuration step'
    echo -e '\nExample calls:'
    echo "$PROGRAM -B build"
    echo "$PROGRAM -B build --gpu"
    echo "$PROGRAM -B build --cxx --with-boost --without-gmp --venv=/tmp/venv"
    echo "$PROGRAM -B build -- -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc"
    echo "$PROGRAM -B build --cxx --gpu -- -DCMAKE_NVCXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/nvc++"
}

# --------------------------------------

# shellcheck disable=SC2034
getopts_args_extra='c'

function parse_extra_args() {
    case "$1" in
        clean )          no_arg;
                         set_var do_clean
                         ;;
        c | configure )  no_arg;
                         set_var do_configure
                         ;;
        configure-only ) no_arg;
                         set_var configure_only
                         ;;
        doc | docs )     no_arg;
                         set_var do_docs
                         ;;
        install)         no_arg;
                         set_var do_install
                         ;;
        prefix)          needs_arg;
                         set_var prefix_dir "$2"
                         ;;
        ??* )            return 2  # Delegate error handling to main argument parsing function
                         ;;
    esac
}

# ------------------------------------------------------------------------------

# NB: using the default values from parse_common_args.sh
. "$ROOTDIR/scripts/build/parse_common_args.sh"

# Locate python or python3
. "$ROOTDIR/scripts/build/locate_python3.sh"

# ==============================================================================

set -e

cd "${ROOTDIR}"

# ------------------------------------------------------------------------------
# Create a virtual environment for building the wheel

if [ "$do_clean_build_dir" -eq 1 ]; then
    echo "Deleting build folder: $build_dir"
    call_cmd rm -rf "$build_dir"
fi

# NB: `created_venv` variable can be used to detect if a virtualenv was created or not
. "$ROOTDIR/scripts/build/python_virtualenv_activate.sh"

if [ "$dry_run" -ne 1 ]; then
    # Make sure the root directory is in the virtualenv PATH
    site_pkg_dir=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')
    pth_file="$site_pkg_dir/mindquantum_local.pth"

    if [ ! -e "$pth_file" ]; then
        echo "Creating pth-file in $pth_file"
        echo "$ROOTDIR" > "$pth_file"
    fi
fi

# ------------------------------------------------------------------------------
# Locate cmake or cmake3

# NB: `cmake_from_venv` variable is set by this script (and is used by python_virtualenv_update.sh)
. "$ROOTDIR/scripts/build/locate_cmake.sh"

# ------------------------------------------------------------------------------
# Locate ninja if needed

if [ "$cmake_generator" == "Ninja" ]; then
    # NB: `ninja_from_venv` variable is set by this script (and is used by python_virtualenv_update.sh)
    . "$ROOTDIR/scripts/build/locate_ninja.sh"
fi

# ------------------------------------------------------------------------------
# Update Python virtualenv (if requested/necessary)

. "$ROOTDIR/scripts/build/python_virtualenv_update.sh"

# ------------------------------------------------------------------------------
# Setup arguments for build

CMAKE_BOOL=(OFF ON)

cmake_args=(-DIN_PLACE_BUILD:BOOL=ON
            -DIS_PYTHON_BUILD:BOOL=OFF
            -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
            -DCMAKE_BUILD_TYPE:STRING="$build_type"
            -DENABLE_ANALYZER:BOOL="${CMAKE_BOOL[$enable_analyzer]}"
            -DENABLE_CMAKE_DEBUG:BOOL="${CMAKE_BOOL[$cmake_debug_mode]}"
            -DENABLE_CUDA:BOOL="${CMAKE_BOOL[$enable_gpu]}"
            -DENABLE_DOCUMENTATION:BOOL="${CMAKE_BOOL[$do_docs]}"
            -DENABLE_GITEE:BOOL="${CMAKE_BOOL[$enable_gitee]}"
            -DENABLE_LOGGING:BOOL="${CMAKE_BOOL[$enable_logging]}"
            -DENABLE_LOGGING_DEBUG_LEVEL:BOOL="${CMAKE_BOOL[$logging_enable_debug]}"
            -DENABLE_LOGGING_TRACE_LEVEL:BOOL="${CMAKE_BOOL[$logging_enable_trace]}"
            -DBUILD_TESTING:BOOL="${CMAKE_BOOL[$enable_tests]}"
            -DCLEAN_3RDPARTY_INSTALL_DIR:BOOL="${CMAKE_BOOL[$do_clean_3rdparty]}"
            -DUSE_VERBOSE_MAKEFILE:BOOL="${CMAKE_BOOL[! $cmake_make_silent]}"
            -DCMAKE_FIND_USE_PACKAGE_REGISTRY:BOOL="${CMAKE_BOOL[! $cmake_no_registry]}"
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY:BOOL="${CMAKE_BOOL[! $cmake_no_registry]}")
make_args=()

if [[ -n "$cmake_generator" ]]; then
    cmake_args+=(-G "${cmake_generator}")
fi

if [[ -n "$prefix_dir" ]]; then
    cmake_args+=(-DCMAKE_INSTALL_PREFIX:FILEPATH="${prefix_dir}")
fi

if [ "$enable_ccache" -eq 1 ]; then
    ccache_exec=""
    if command -v ccache > /dev/null 2>&1; then
        ccache_exec=ccache
    elif command -v sccache > /dev/null 2>&1; then
        ccache_exec=sccache
    fi
    if [ -n "$ccache_exec" ]; then
        ccache_exec=$(which "$ccache_exec")
        cmake_args+=(-DCMAKE_C_COMPILER_LAUNCHER="$ccache_exec")
        cmake_args+=(-DCMAKE_CXX_COMPILER_LAUNCHER="$ccache_exec")
        if [ "$enable_gpu" -eq 1 ]; then
            cmake_args+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$ccache_exec")
        fi
    fi
fi

if [[ "$enable_gpu" -eq 1 && -n "$cuda_arch" ]]; then
    cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES:STRING="$cuda_arch")
fi

local_pkgs_str=$(join_by , "${local_pkgs[@]}")
if [[ "$force_local_pkgs" -eq 1 ]]; then
    cmake_args+=(-DMQ_FORCE_LOCAL_PKGS=all)
elif [ -n "$local_pkgs_str" ]; then
    cmake_args+=(-DMQ_FORCE_LOCAL_PKGS="$local_pkgs_str")
fi

if [ "$n_jobs" -ne -1 ]; then
    cmake_args+=(-DJOBS:STRING="$n_jobs")
    make_args+=(-j "$n_jobs")
fi

# NB: CMake < 3.24 typically set CC, CXX during the first run, which basically overwrites the values in CC, CXX. In
#     order to work around that, we explicitly set the compilers using the related CMake variables.
if [ -n "$CC" ]; then
    cmake_args+=(-DCMAKE_C_COMPILER:FILEPATH="$CC")
fi
if [ -n "$CXX" ]; then
    cmake_args+=(-DCMAKE_CXX_COMPILER:FILEPATH="$CXX")
fi
if [ -n "$CUDACXX" ]; then
    cmake_args+=(-DCMAKE_CUDA_COMPILER:FILEPATH="$CUDACXX")
fi

# ------------------------------------------------------------------------------

if [ "$enable_gpu" -eq 1 ]; then
    # Older CMake using find_package(CUDA) would rely on CUDA_HOME, but newer CMake only look at CUDACXX and CUDA_PATH
    if [[ -n "$CUDA_HOME" && -z "$CUDA_PATH" ]]; then
        echo 'CUDA_HOME is defined, but CUDA_PATH is not. Setting CUDA_PATH=CUDA_HOME'
        export CUDA_PATH="$CUDA_HOME"
    fi

    debug_print "CUDA_PATH = $CUDA_PATH"
fi

# ------------------------------------------------------------------------------
# Build

if [[ ! -d "$build_dir" || "$do_clean_build_dir" -eq 1 ]]; then
    do_configure=1
elif [ "$do_clean_cache" -eq 1 ]; then
    do_configure=1
    echo "Removing CMake cache at: $build_dir/CMakeCache.txt"
    call_cmd rm -f "$build_dir/CMakeCache.txt"
    echo "Removing CMake files at: $build_dir/CMakeFiles"
    call_cmd  rm -rf "$build_dir/CMakeFiles"
    echo "Removing CMake files at: $build_dir/cmake-ldtest*"
    call_cmd  rm -rf "$build_dir/cmake-ldtest*"
fi

if [ "$do_configure" -eq 1 ]; then
    call_cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}" "$@"
fi

if [ "$configure_only" -eq 1 ]; then
    exit 0
fi

target=all
if [ "$do_install" -eq 1 ]; then
    target=install
fi

if [ "$do_clean" -eq 1 ]; then
    call_cmake --build "$build_dir" --target clean
fi

if [ "$do_docs" -eq 1 ]; then
    call_cmake --build "$build_dir" --target docs --config "$build_type" "${make_args[@]}"
fi

call_cmake --build "$build_dir" --config "$build_type" --target "${target}" "${make_args[@]}"

# ==============================================================================
