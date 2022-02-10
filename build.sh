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
OUTPUT_PATH="${BASEPATH}/output"
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

$PYTHON -m venv venv
source venv/bin/activate
$PYTHON -m pip install -U pip setuptools build pybind11

# ------------------------------------------------------------------------------
# Setup arguments for build

args=(--set ENABLE_PROJECTQ --unset ENABLE_QUEST)

if [[ $1 = "gpu" ]]; then
    args+=(--set ENABLE_CUDA --unset MULTITHREADED --set VERBOSE_CMAKE)
fi

fixed_args=()
for arg in "${args[@]}"; do
    fixed_args+=("-C--global-option=$arg")
done

# ------------------------------------------------------------------------------
# Build the wheels

echo ${PYTHON} -m build -w "${fixed_args[*]}"
${PYTHON} -m build -w "${fixed_args[@]}" "$@"

# ------------------------------------------------------------------------------
# Move the wheels to the output directory

mk_new_dir "${OUTPUT_PATH}"
mv -v dist/* "${OUTPUT_PATH}"


echo "------Successfully created mindquantum package------"
