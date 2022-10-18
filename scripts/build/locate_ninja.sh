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

[ "${_sourced_locate_ninja}" != "" ] && return || _sourced_locate_ninja=.

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

has_ninja=0
ninja_from_venv=0  # mainly useful if updating the virtualenv

if [ -f "$python_venv_path/bin/ninja" ]; then
    NINJA="$python_venv_path/bin/ninja"
    has_ninja=1
    # shellcheck disable=SC2034
    ninja_from_venv=1
elif [ -f "$python_venv_path/Scripts/ninja" ]; then
    NINJA="$python_venv_path/Scripts/ninja"
    has_ninja=1
    # shellcheck disable=SC2034
    ninja_from_venv=1
fi

# ------------------------------------------------------------------------------

if [ $has_ninja -ne 1 ]; then
    if command -v ninja > /dev/null 2>&1; then
        NINJA=ninja
        has_ninja=1
    fi
fi

# ==============================================================================

pip_args=()
if [ "$_IS_MINDSPORE_CI" -eq 1 ]; then
    pip_args+=(-i https://mirror.baidu.com/pypi/simple)
fi

if [ $has_ninja -eq 0 ]; then
    echo "Installing Ninja inside the Python virtual environment"
    call_cmd "$PYTHON" -m pip install -U pip
    call_cmd "$PYTHON" -m pip install "${pip_args[@]}" ninja
    # shellcheck disable=SC2034
    NINJA="$python_venv_path/bin/ninja"
fi

#==============================================================================

unset has_ninja
