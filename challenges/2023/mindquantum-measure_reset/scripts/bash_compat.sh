#! /bin/bash

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

[ "${_sourced_bash_compat}" != "" ] && return || _sourced_bash_compat=.

# ==============================================================================

if ! command -v realpath > /dev/null 2>&1; then
    function realpath() {
        pushd "$(dirname "$1")" > /dev/null 2>&1 || exit 1
        LINK=$(readlink "$(basename "$1")")
        while [ "$LINK" ]; do
            cd "$(dirname "$LINK")"  || exit 1
            LINK=$(readlink "$(basename "$1")")
        done
        REALPATH="$PWD/$(basename "$1")"
        popd > /dev/null 2>&1 || exit 1
        echo "$REALPATH"
    }
fi

# ==============================================================================

function declare_var() {
    local name value
    name="$1" && shift
    value="$1" && shift

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "${name}=${value}"
    else
        declare -g "${name}=${value}"
    fi
}

# ==============================================================================

function declare_int() {
    local name value
    name="$1" && shift
    value="$1" && shift

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "${name}=${value}"
    else
        declare -gi "${name}=${value}"
    fi
}

# ------------------------------------------------------------------------------

function declare_bool_false() {
    declare_int "$1" 0
}

function declare_bool_true() {
    declare_int "$1" 1
}

# ==============================================================================

function declare_A() {
    local array
    array="$1" && shift
    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "${array}=()"
    else
        declare -ga "$array"
    fi
}


# ==============================================================================

function declare_AA() {
    local array
    array="$1" && shift
    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "${array}_keys="
    else
        declare -gA "$array"
    fi
}

# ------------------------------------------------------------------------------

function set_AA() {
    local array key value cur_value
    array="$1" && shift
    key="$1" && shift
    value="$1" && shift

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "${array}_${key}='$value'"
        cur_value=$(eval "echo \${${array}_keys}")
        if [ -z "$cur_value" ]; then
            eval "${array}_keys='$key'"
        else
            eval "${array}_keys='$cur_value $key'"
        fi
    else
        eval "${array}[$key]='$value'"
    fi
}

function get_AA_keys() {
    local array
    array="$1" && shift
    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "echo \${${array}_keys}"
    else
        eval "echo \${!${array}[*]}"
    fi
}

function get_AA_value() {
    local array key

    array="$1" && shift
    key="$1" && shift

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval "echo \${${array}_${key}}"
    else
        eval "echo \${${array}[${key}]}"
    fi
}
