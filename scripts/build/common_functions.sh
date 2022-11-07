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

# shellcheck disable=SC2154,SC2034

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

[ "${_sourced_common_functions}" != "" ] && return || _sourced_common_functions=.

# ==============================================================================

# shellcheck source=SCRIPTDIR/../bash_compat.sh
. "$BASEPATH/../bash_compat.sh"

# shellcheck source=SCRIPTDIR/../parse_ini.sh
. "$BASEPATH/../parse_ini.sh"

# ==============================================================================

ncolors=$(tput colors 2> /dev/null)
if [ -n "$ncolors" ] && [ "$ncolors" -ge 16 ]; then
    _BOLD="$(tput bold)"
    _UNDERLINE="$(tput smul)"
    _STANDOUT="$(tput smso)"
    _NORMAL="$(tput sgr0)"
    _BLACK="$(tput setaf 0)"
    _RED="$(tput setaf 1)"
    _GREEN="$(tput setaf 2)"
    _YELLOW="$(tput setaf 3)"
    _BLUE="$(tput setaf 4)"
    _MAGENTA="$(tput setaf 5)"
    _CYAN="$(tput setaf 6)"
    _WHITE="$(tput setaf 7)"
    _GREY="$(tput setaf 8)"
fi
unset ncolors

# ==============================================================================

function check_for_verbose() {
    for arg in "$@"; do
        if [[ "$arg" == '-v' || $arg == '--verbose' ]]; then
            # shellcheck disable=SC2034
            verbose=1
            break
        fi
    done
}

# ==============================================================================

function debug_print() {
    if [ "${verbose:-0}" -eq 1 ]; then
        echo "${_YELLOW}DEBUG $*${_NORMAL}" >&2
    fi
}

function print_warning() {
    echo "${_GREEN}**********${_NORMAL}" >&2
    echo "${_STANDOUT}${_GREEN}WARN $*${_NORMAL}" >&2
    echo "${_GREEN}**********${_NORMAL}" >&2
}

# ------------------------------------------------------------------------------

function assign_value() {
    name=$1
    shift
    value=$1
    shift
    output_only=${1:-0}

    eval_str="$name"

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        value_lower=$(echo "$value" | tr "[:upper:]" "[:lower:]")
    else
        value_lower=${value,,}
    fi

    if [[ ${value_lower} =~ ^(yes|true)$ ]]; then
        eval_str="$eval_str=1"
    elif [[ ${value_lower} =~ ^(no|false)$ ]]; then
        eval_str="$eval_str=0"
    elif [[ ${value_lower} =~ ^[0-9]+$ ]]; then
        eval_str="$eval_str=$value"
    elif [[ ${value_lower} =~ \"\ \" ]]; then
        eval_str="$eval_str=( $value )"
    else
        eval_str="$eval_str=\"$value\""
    fi

    if [ "$output_only " -eq 1 ]; then
        echo "$eval_str"
    else
        debug_print "$eval_str"
        eval "$eval_str"
    fi
}

function set_var() {
    local name value

    name=$1
    shift
    value=${1:-1}

    assign_value "$name" "$value"
    assign_value "_${name}_was_set" 1
}

function die() {
    # complain to STDERR and exit with error
    echo "${_BOLD}${_RED}$*${_NORMAL}" >&2; exit 2;
}

function no_arg() {
    if [ -n "$OPTARG" ]; then
        die "No arg allowed for --$OPT option";
    fi
}

function needs_arg() {
    if [ -z "$OPTARG" ]; then
        die "No arg for --$OPT option"
    fi
    if [ "$flag_value" -eq 0 ]; then
        die "Cannot specify --no-$OPT for non-flag argument --$OPT"
    fi
}

# ------------------------------------------------------------------------------

function is_abspath {
    case "x$1" in
        (x*/..|x*/../*|x../*|x*/.|x*/./*|x./*)
            rc=1
            ;;
        (x/*)
            rc=0
            ;;
        (*)
            rc=1
            ;;
    esac
    return $rc
}

# __set_variable_from_ini <section> <check_set> <check_null> <dry_run>
function __set_variable_from_ini {
    local section check_set check_null do_dry_run var value eval_str null_test root_dir
    section=$1 && shift
    check_set=$1 && shift
    check_null=$1 && shift
    do_dry_run=$1 && shift
    root_dir=$(realpath "$ROOTDIR")

    for var in $(get_AA_keys "configuration_${section/./_}"); do
        value=$(get_AA_value "configuration_${section/./_}" "$var")

        if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
            value_lower=$(echo "$value" | tr "[:upper:]" "[:lower:]")
        else
            value_lower=${value,,}
        fi

        # Remove trailing comments and trim value string
        if [ "$(uname)" == "Darwin" ]; then
            if command -v gsed >/dev/null 2>&1; then
                value_lower=$(echo "$value_lower" | gsed -e 's/\s*#.*$//;s/^[ \t]*//;s/[ \t]*$//')
            else
                # The default sed on macOS does not play well with '\t'
                value_lower=$(echo "$value_lower" | sed -E 's/\s*#.*$//g;s/^[ ]*//g;s/[ ]*$//g')
            fi
        else
            value_lower=$(echo "$value_lower" | sed -e 's/\s*#.*$//;s/^[ \t]*//;s/[ \t]*$//')
        fi

        eval_str=''
        # shellcheck disable=SC2016
        null_test='-z "${%s}"'

        if [[ $section =~ ^.*(path|paths)$ ]]; then
            if [ -n "$value" ]; then
                if ! is_abspath "$value"; then
                    value="\"$root_dir/$value\""
                fi
            fi
            eval_str="declare_var $var $value"
            # NB: second part of the condition below is an attempt at compatibility with older BASH
        elif [[ ${value_lower} =~ ^(true|yes)$ || "${value_lower}" == "true" ]]; then
            eval_str="declare_bool_true $var"
            # NB: second part of the condition below is an attempt at compatibility with older BASH
        elif [[ ${value_lower} =~ ^(false|no)$ || "${value_lower}" == "false" ]]; then
            eval_str="declare_bool_false $var"
        elif [[ ${value} =~ ^.*\|.*$ ]]; then
            if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
                null_test='%s_keys'
                eval_str="declare_A $var"
                IFS_old="$IFS"
                IFS='|'
                for el in $value; do
                    eval_str="$eval_str && $var+=(${el})"
                done
                IFS="$IFS_old"
            else
                if [ "${BASH_VERSINFO[0]}" -lt 5 ]; then
                    # shellcheck disable=SC2016
                    null_test='-v "${%s}"'
                else
                    # shellcheck disable=SC2016
                    null_test='"${%s@a}"'
                fi
                eval_str="declare_A $var && mapfile -t $var <<< \"$(echo -e "${value/|/\\n}")\""
            fi
        elif [[ $value =~ ^-?[0-9]+$ ]]; then
            eval_str="declare_int $var $value"
        else
            eval_str="declare_var $var '$value'"
        fi

        debug_print "$(printf "%-60s  # [%s]" "$eval_str" "$section")"

        if [ "$check_set" -eq 1 ]; then
            eval_str=$(printf "[[ \"\${_%s_was_set:-0}\" -eq 0 ]] && %s" "$var" "$eval_str")
        elif [ "$check_null" -eq 1 ]; then
            eval_str=$(printf "[[ ${null_test} ]] && %s" "$var" "$eval_str")
        fi

        if [ "$do_dry_run" -eq 0 ]; then
            # debug_print "  invoked expression: $eval_str"
            eval "$eval_str"
        fi
    done
}

# --------------------------------------

# set_variable_from_ini <filename> [-s <section>] [-c] [-C] [-n]
# -c: check_set
# -C: check_null
# -n: dry_run
function set_variable_from_ini {
    local target_section do_dry_run check_set check_null OPT OPTARG OPTIND

    declare -i do_dry_run=0
    declare -i check_set=0
    declare -i check_null=0
    target_section=''
    while getopts "s:cCn" OPT; do
        case "$OPT" in
            s)  target_section="$OPTARG"
                ;;
            c)  check_set=1
                ;;
            C)  check_null=1
                ;;
            n)  do_dry_run=1
                ;;
            /?) exit 2
            ;;
        esac
    done
    shift $((OPTIND-1)) # remove parsed options and args from $@ list

    parse_ini_file "$1"

    if [ -n "$target_section" ]; then
        __set_variable_from_ini "$target_section" $check_set $check_null $do_dry_run
    else
        for section in "${configuration_sections[@]}"; do
            __set_variable_from_ini "$section" $check_set $check_null $do_dry_run
        done
    fi
}

# ==============================================================================

call_cmd() {
    if [ "${dry_run:-0}" -ne 1 ]; then
        debug_print "Calling command: $*"
        if ! "$@"; then
            die "Command failed: $*"
        fi
        return 0
    else
        echo "$@"
        return 0
    fi
}

# ------------------------------------------------------------------------------

call_cmake() {
    if [ "${dry_run:-0}" -ne 1 ]; then
        echo "**********"
        echo "Calling CMake with: cmake " "$@"
        echo "**********"
    fi
    call_cmd "$CMAKE" "$@"
    return $?
}

# ==============================================================================

function join_by {
    local d=${1-} f=${2-}
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}

# ==============================================================================

function version_less_equal() {
    debug_print "  comparing version '$1' with '$2'"
    local a_major a_minor b_major b_minor
    a_major=$(echo "$1" | cut -d. -f1)
    a_minor=$(echo "$1" | cut -d. -f2)
    b_major=$(echo "$2" | cut -d. -f1)
    b_minor=$(echo "$2" | cut -d. -f2)

    if [ "$a_major" -le "$b_major" ]; then
        if [ "$a_minor" -le "$b_minor" ]; then
            return 0
        fi
    fi
    return 1
}

# ==============================================================================
