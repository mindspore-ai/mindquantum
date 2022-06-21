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

[ "${_sourced_locate_python3}" != "" ] && return || _sourced_locate_python3=.

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

# ------------------------------------------------------------------------------

# shellcheck source=SCRIPTDIR/default_values.sh
. "$BASEPATH/default_values.sh"

# shellcheck source=SCRIPTDIR/common_functions.sh
. "$BASEPATH/common_functions.sh"

# ==============================================================================

if [ -z "$PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo 'Unable to locate python or python3!' 1>&2
        exit 1
    fi

    debug_print "Using Python from: $(which $PYTHON)"
elif [ "${verbose:-0}" -eq 1 ]; then
    debug_print "Using Python from environment variable: $(which $PYTHON)"
fi

# ==============================================================================
