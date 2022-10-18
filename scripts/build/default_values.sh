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

[ "${_sourced_default_values}" != "" ] && return || _sourced_default_values=.

# ==============================================================================
# Default values for input arguments

if [ -z "$n_jobs_default" ]; then
    if command -v nproc >/dev/null 2>&1; then
        n_jobs_default=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        n_jobs_default=$(sysctl -n hw.logicalcpu)
    else
        n_jobs_default=8
    fi
fi

# ==============================================================================

: "${third_party_libraries=$(cd "$ROOTDIR/third_party" \
                                && find . -maxdepth 1 -type d ! -path . | grep -vE '(cmake|CMakeLists.txt)' \
                                    | sed 's|./||')}"

# ==============================================================================
# Other helper variables

: "${cmake_from_venv=0}"
: "${ninja_from_venv=0}"
