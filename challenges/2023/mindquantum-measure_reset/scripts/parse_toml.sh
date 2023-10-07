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

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

[ "${_sourced_parse_toml}" != "" ] && return || _sourced_parse_toml=.

# ==============================================================================

# shellcheck source=SCRIPTDIR/bash_compat.sh
. "$BASEPATH/bash_compat.sh"

# ==============================================================================

if [ -z "$SED" ]; then
    if command -v gsed >/dev/null 2>&1; then
        SED='gsed'
    elif command -v awk >/dev/null 2>&1; then
        SED='sed'
    else
        echo 'ERROR: Unable to locate gsed or sed!' 1>&2
    fi
fi

# ==============================================================================

function read_build_system_requires() {
    if [ -z "$SED" ]; then
        return
    fi
    local filename="$1"

    debug_print "Reading 'build-system.requires' from TOML file: $filename"

    packages=$($SED -e 's/requires *= */requires = /' "$filename" \
                   | $SED -n '/requires =/,/^]$/p' \
                   | $SED -e '/requires =/d' -e '/^]$/d' -e "s/'//g" -e 's/^ *//g' -e 's/,$//' \
                   | tr '\n' ' ')

    if [ -z "$packages" ]; then
        die "Failed to parse [build-system.requires] from $filename"
    fi

    declare_A build_requires
    for pkg in $packages; do
        build_requires+=( "$pkg" )
    done

    debug_print "  read ${build_requires[*]}"
}

# ------------------------------------------------------------------------------

function read_project_dependencies() {
    if [ -z "$SED" ]; then
        return
    fi
    local filename="$1"

    debug_print "Reading 'project.dependencies' from TOML file: $filename"

    packages=$($SED -n '/dependencies =/,/^]$/p' "$filename" \
                   | $SED -e '/dependencies =/d' -e '/^]$/d' -e "s/'//g" -e 's/^ *//g' -e 's/,$//' \
                   | tr '\n' ' ')

    declare_A dependencies
    for pkg in $packages; do
        dependencies+=( "$pkg" )
    done

    debug_print "  read ${dependencies[*]}"
}

# ==============================================================================
