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

[ "${_sourced_parse_common_args}" != "" ] && return || _sourced_parse_common_args=.

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOTDIR="$BASEPATH/../.."
PROGRAM=$(basename "$0")

# ==============================================================================

if [ -z "$ROOTDIR" ]; then
    die '(internal error): ROOTDIR variable not defined!'
fi

# ==============================================================================

: "${config_file=$ROOTDIR/build.conf}"

# shellcheck source=SCRIPTDIR/common_functions.sh
. "$BASEPATH/common_functions.sh"

check_for_verbose "$@"

# Read default values from default config file (without overriding any defined Bash variables)
echo "Reading INI/Unix default configuration"
set_variable_from_ini -C "$BASEPATH/default_values.conf"

# Other default values (ie. those not present in the config file)
# shellcheck source=SCRIPTDIR/default_values.sh
. "$BASEPATH/default_values.sh"

# ==============================================================================

# NB: here since the `realpath` function might not exist
ROOTDIR=$(realpath "$BASEPATH/../..")

# ==============================================================================

print_show_libraries() {
    echo 'Known third-party libraries:'
    # shellcheck disable=SC2034
    for lib in $third_party_libraries; do
        echo " - $lib"
    done
}

parse_with_libraries() {
    if ! echo "$third_party_libraries" | tr ' ' '\n' | grep -F -x -q "$1" ; then
        print_show_libraries
        echo ''
        die "Unknown library for --with-$1 or --without-$1"
    fi

    if [ "$1" == "projectq" ]; then
        set_var enable_projectq "$2"
    elif [ "$2" -eq 1 ]; then
        local_pkgs+=("$1")
        declare_bool_true _local_pkgs_was_set
    else
        for index in "${!local_pkgs[@]}" ; do [[ ${local_pkgs[$index]} == "$1" ]] && unset -v 'local_pkgs[$index]' ; done
        local_pkgs=("${local_pkgs[@]}")
        declare_bool_true _local_pkgs_was_set
    fi
}

# ------------------------------------------------------------------------------

help_message() {
    help_header

    echo -e '\nUsage:'
    echo "  $PROGRAM [options] [-- cmake_options]"
    echo -e '\nOptions:'
    echo '  -h,--help              Show this help message and exit'
    echo '  -n                     Dry run; only print commands but do not execute them'
    echo ''
    echo '  -B,--build=[dir]       Specify build directory'
    echo "                         Defaults to: $build_dir"
    echo '  --analyzer             Use the compiler static analysis tool during compilation (GCC & MSVC)'
    echo '  --ccache               If ccache or sccache are found within the PATH, use them with CMake'
    echo '  --clean-3rdparty       Clean 3rd party installation directory'
    echo '  --clean-all            Clean everything before building.'
    echo '                         Equivalent to --clean-venv --clean-builddir'
    echo '  --clean-builddir       Delete build directory before building'
    echo '  --clean-cache          Re-run CMake with a clean CMake cache'
    echo '  --clean-venv           Delete Python virtualenv before building'
    echo '  --cmake-no-registry    Do not use the CMake registry to find packages'
    echo '  --config=[dir]         Path to INI configuration file with default values for the parameters'
    echo "                         Defaults to: $config_file"
    echo '                         NB: command line arguments always take precedence over configuration file values'
    echo '  --cxx                  (experimental) Enable MindQuantum C++ support'
    echo '  --debug                Build in debug mode'
    echo '  --debug-cmake          Enable debugging mode for CMake configuration step'
    echo '  --gitee                Use Gitee (where possible) instead of Github/Gitlab'
    echo '  --no-gitee             Do not favor Gitee over Github/Gitlab'
    echo '  --gpu                  Enable GPU support'
    echo '  -j,--jobs [N]          Number of parallel jobs for building'
    echo "                         Defaults to: $n_jobs_default"
    echo '  --local-pkgs           Compile third-party dependencies locally'
    echo '  --logging              Enable logging in C++ code'
    echo '  --logging-debug        Enable DEBUG level logging macros (implies --logging)'
    echo '  --logging-trace        Enable TRACE level logging macros (implies --logging --logging-debug)'
    echo '  --no-config            Ignore any configuration file'
    echo '  --ninja                Build using Ninja instead of make'
    echo '  --quiet                Disable verbose build rules'
    echo '  --show-libraries       Show all known third-party libraries'
    echo '  -v, --verbose          Enable verbose output from the Bash scripts'
    echo '  --venv=[dir]           Path to Python virtual environment'
    echo "                         Defaults to: $python_venv_path"
    echo '  --with-<library>       Build the third-party <library> from source'
    echo '                         (ignored if --local-pkgs is passed, except for projectq)'
    echo '  --without-<library>    Do not build the third-party library from source'
    echo '                         (ignored if --local-pkgs is passed, except for projectq)'
    echo ''
    echo 'You may negate any flag argument (ie. arguments that do not require a value) by prefixing them with "--no-"'
    echo 'e.g. --no-logging to disable logging'
    echo ''
    echo 'Test related options:'
    echo '  --test                 Build C++ tests and install dependencies for Python testing as well'
    echo '  --only-pytest          Only install pytest and its dependencies when creating/building the virtualenv'
    echo ''
    echo 'CUDA related options:'
    echo '  --cuda-arch=[arch]     Comma-separated list of architectures to generate device code for.'
    echo '                         Only useful if --gpu is passed. See CMAKE_CUDA_ARCHITECTURES for more information.'
    echo ''
    echo 'Python related options:'
    echo '  --update-venv          Update the python virtual environment'
    echo ''
    echo 'Developer options:'
    echo '  --cmake-no-registry    Disable the use of CMake package registries during configuration'
    echo ''

    if command -v extra_help >/dev/null 2>&1; then
        extra_help
    fi
}

# ==============================================================================

: "${has_extra_args=0}"
getopts_args='B:hnvj:-:'

if [ -n "$getopts_args_extra" ]; then
    has_extra_args=1
    getopts_args="${getopts_args_extra}${getopts_args}"

    if ! command -v parse_extra_args >/dev/null 2>&1; then
        die "Must provide a function named 'parse_extra_args' since 'getopts_args_extra' is defined."
    fi
fi

while getopts "${getopts_args}" OPT; do
    # shellcheck disable=SC2214,SC2295
    if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi

    if [[ $OPT =~ with-([a-zA-Z0-9_]+) ]]; then
        OPT=with
        enable_lib=1
        library=${BASH_REMATCH[1]}
    elif [[ $OPT =~ without-([a-zA-Z0-9_]+) ]]; then
        OPT=with
        enable_lib=0
        library=${BASH_REMATCH[1]}
    fi

    flag_value=1
    if [[ ! $OPT == "no-config" && $OPT =~ ^no-([a-zA-Z0-9_-]+) ]]; then
        OPT="${BASH_REMATCH[1]}"
        flag_value=0
    fi

    case "$OPT" in
        h | help )          no_arg;
                            help_message >&2
                            exit 1 ;;
        B | build)          needs_arg;
                            # shellcheck disable=SC2034
                            set_var build_dir "$OPTARG"
                            ;;
        analyzer )          no_arg;
                            set_var enable_analyzer $flag_value
                            ;;
        ccache )            no_arg;
                            set_var enable_ccache $flag_value
                            ;;
        cmake-no-registry ) no_arg;
                            set_var cmake_no_registry $flag_value
                            ;;
        config )            needs_arg;
                            set_var config_file "$OPTARG"
                            ;;
        clean-3rdparty )    no_arg;
                            set_var do_clean_3rdparty $flag_value
                            ;;
        clean-all )         no_arg;
                            set_var do_clean_venv $flag_value
                            set_var do_clean_build_dir $flag_value
                            ;;
        clean-builddir )    no_arg;
                            set_var do_clean_build_dir $flag_value
                            ;;
        clean-cache )       no_arg
                            set_var do_clean_cache $flag_value
                            ;;
        clean-venv )        no_arg;
                            set_var do_clean_venv $flag_value
                            ;;
        cuda-arch )         needs_arg;
                            set_var cuda_arch "$(echo "$OPTARG" | tr ',' ';')"
                            ;;
        cxx )               no_arg;
                            set_var enable_cxx $flag_value
                            ;;
        debug )             no_arg;
                            set_var build_type 'Debug'
                            ;;
        debug-cmake )       no_arg;
                            set_var cmake_debug_mode $flag_value
                             ;;
        gitee )             no_arg;
                            set_var enable_gitee $flag_value
                            ;;
        gpu )               no_arg;
                            set_var enable_gpu $flag_value
                            ;;
        j | jobs )          needs_arg;
                            set_var n_jobs "$OPTARG"
                            ;;
        local-pkgs )        no_arg;
                            set_var force_local_pkgs $flag_value
                            ;;
        logging )           no_arg;
                            set_var enable_logging $flag_value
                            ;;
        logging-debug )     no_arg;
                            set_var enable_logging $flag_value
                            set_var logging_enable_debug $flag_value
                            ;;
        logging-trace )     no_arg;
                            set_var logging_enable_debug $flag_value
                            set_var logging_enable_trace $flag_value
                            ;;
        n )                 no_arg;
                            set_var dry_run $flag_value
                            ;;
        no-config )         no_arg;
                            set_var config_file '__disabled_config__'
                            ;;
        ninja )             no_arg;
                            set_var cmake_generator 'Ninja Multi-Config'
                            ;;
        only-pytest )       no_arg;
                            set_var only_install_pytest $flag_value
                            ;;
        quiet )             no_arg;
                            set_var cmake_make_silent $flag_value
                            ;;
        show-libraries )    no_arg;
                            print_show_libraries
                            exit 1
                            ;;
        test )              no_arg;
                            set_var enable_tests $flag_value
                            ;;
        update-venv )       no_arg;
                            set_var do_update_venv $flag_value
                            ;;
        v | verbose )       no_arg;
                            set_var verbose $flag_value
                            ;;
        venv )              needs_arg;
                            set_var python_venv_path "$OPTARG"
                            ;;
        with )              no_arg;
                            parse_with_libraries "$library" $enable_lib
                            ;;
        \? )                # bad short option (error reported via getopts)
                            exit 2
                            ;;
        * )                 success=0
                            if [ $has_extra_args -eq 1 ]; then
                                parse_extra_args "$OPT" "$OPTARG" "$flag_value"
                                success="$?"
                            fi
                            if [ $success -ne 0 ]; then
                                if [ $flag_value -eq 1 ]; then
                                    die "Illegal option: $OPT or --$OPT"
                                else
                                    # NB: we were looking for --no-$OPT
                                    die "No option found for $OPT or --$OPT (received --no-$OPT on cmdline)"
                                fi
                            fi
                            ;;
    esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# ==============================================================================

if [ -f "$config_file" ]; then
    config_file=$(realpath "$config_file")
    echo "Reading INI/Unix conf configuration file: $config_file"
    debug_print 'NB: overriding values only if not specified on the command line'

    # NB: Check whether the variables were set from the command line and do not override those
    set_variable_from_ini -c "$config_file"
else
    debug_print "Configuration file ($config_file) ignored because file does not exist"
fi

if [[ $n_jobs -eq -1 && ! $cmake_generator == "Ninja"  ]]; then
    n_jobs=$n_jobs_default
fi

# ==============================================================================
