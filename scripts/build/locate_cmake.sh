#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

[ "${_sourced_locate_cmake}" != "" ] && return || _sourced_locate_cmake=.

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ------------------------------------------------------------------------------

# shellcheck source=SCRIPTDIR/default_values.sh
. "$BASEPATH/default_values.sh"

# shellcheck source=SCRIPTDIR/common_functions.sh
. "$BASEPATH/common_functions.sh"

# ==============================================================================

if [ -z "$ROOTDIR" ]; then
    die '(internal error): ROOTDIR variable not defined!'
fi
if [ -z "$PYTHON" ]; then
    die '(internal error): PYTHON variable not defined!'
fi
if [ -z "$python_venv_path" ]; then
    die '(internal error): python_venv_path variable not defined!'
fi

# ==============================================================================

has_cmake=0
cmake_from_venv=0  # mainly useful if updating the virtualenv

if [ -f "$python_venv_path/bin/cmake" ]; then
    CMAKE="$python_venv_path/bin/cmake"
    has_cmake=1
    # shellcheck disable=SC2034
    cmake_from_venv=1
elif [ -f "$python_venv_path/Scripts/cmake" ]; then
    CMAKE="$python_venv_path/Scripts/cmake"
    has_cmake=1
    # shellcheck disable=SC2034
    cmake_from_venv=1
fi

# ==============================================================================

cmake_version_min=$(tr -d '\r' < CMakeLists.txt | grep -oE 'cmake_minimum_required\(VERSION\s+[0-9\.]+\s*\)' \
                        | grep -oE '[0-9.]+')

if [ $has_cmake -ne 1 ]; then
    if command -v cmake > /dev/null 2>&1; then
        CMAKE=cmake
    elif command -v cmake3 > /dev/null 2>&1; then
        CMAKE=cmake3
    fi

    if [[ -n "$CMAKE" ]]; then
        cmake_version=$("$CMAKE" --version | head -1 | cut -d' ' -f3)

        if version_less_equal "$cmake_version_min" "$cmake_version"; then
            has_cmake=1
        fi
    fi
fi

pip_args=(--prefer-binary)
if [ "$_IS_MINDSPORE_CI" -eq 1 ]; then
    pip_args+=(-i https://mirror.baidu.com/pypi/simple)
fi

if [ $has_cmake -eq 0 ]; then
    echo "Installing CMake inside the Python virtual environment"
    call_cmd "$PYTHON" -m pip install -U pip
    call_cmd "$PYTHON" -m pip install "${pip_args[@]}" "cmake>=$cmake_version_min"
    CMAKE="$python_venv_path/bin/cmake"
fi

#==============================================================================

unset has_cmake
