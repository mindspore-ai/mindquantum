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

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
ROOTDIR="$BASEPATH/.."

echo "Called with: $*"

# Load common bash helper functions
. "$ROOTDIR/scripts/build/common_functions.sh"

# ==============================================================================

LD_PATH_VAR=DYLD_LIBRARY_PATH

# ------------------------------------------------------------------------------

function usage() {
    echo "Usage: $0 <wheel-filename>"
}

# ==============================================================================

if [ $# -lt 1 ]; then
    usage
    echo 'ERROR: Missing argument: <wheel-filename>' 1>&2
    exit 1
fi

wheel_filename="$1"
shift

if [ ! -r "$wheel_filename" ]; then
    echo "ERROR: wheel filename is not valid: $wheel_filename" 1>&2
    exit 1
fi

# ------------------------------------------------------------------------------

tmp_dir="$(mktemp -d)"
echo "Unzipping wheel in $tmp_dir"
unzip -q -d "$tmp_dir" "$wheel_filename"

# ------------------------------------------------------------------------------
# Update XXX_LIBRARY_PATH

lib_dir_list=()
for dir in "$tmp_dir/mindquantum/lib/mindquantum/third_party/"*; do
    dir="$dir/lib"
    if [ -d "$dir" ]; then
        echo "  prepending to $LD_PATH_VAR: $dir"
        lib_dir_list+=("$dir")
    fi
done
if [ -n "${!LD_PATH_VAR}" ]; then
    lib_dir_list+=("${!LD_PATH_VAR}")
fi

# ------------------------------------------------------------------------------
# Delocate the wheel again (in-place)

export "${LD_PATH_VAR}"="$(join_by : "${lib_dir_list[@]}")"
env_vars=("${LD_PATH_VAR}=${!LD_PATH_VAR}")

delocate_args=(--verbose --check-archs --dylibs-only --require-archs="$(uname -m)")

echo "Calling delocate-wheel ${delocate_args[*]} <wheel>"
env "${env_vars[@]}" delocate-wheel "${delocate_args[@]}" "$wheel_filename"

# ------------------------------------------------------------------------------
# Cleanup

rm -rf "$tmp_dir"

# ==============================================================================
