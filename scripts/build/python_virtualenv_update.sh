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

[ "${_sourced_python_virtualenv_update}" != "" ] && return || _sourced_python_virtualenv_update=.

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

# ------------------------------------------------------------------------------

# shellcheck source=SCRIPTDIR/default_values.sh
. "$BASEPATH/default_values.sh"

# shellcheck source=SCRIPTDIR/common_functions.sh
. "$BASEPATH/common_functions.sh"

# shellcheck source=SCRIPTDIR/../parse_toml.sh
. "$BASEPATH/../parse_toml.sh"

# ==============================================================================

if [ -z "$PYTHON" ]; then
    die '(internal error): PYTHON variable not defined!'
fi

if [ -z "$ROOTDIR" ]; then
    die '(internal error): ROOTDIR variable not defined!'
fi

# ==============================================================================

openblas_python_pkg='scipy-openblas32==0.3.31.22.1'

function query_openblas_info() {
    "$PYTHON" "$ROOTDIR/scripts/build/locate_openblas.py"
}

function export_openblas_info() {
    local openblas_info=$1
    while IFS='=' read -r key value; do
        [ -n "$key" ] || continue
        export "$key=$value"
    done <<< "$openblas_info"
}

function configure_openblas() {
    if [ "${dry_run:-0}" -eq 1 ]; then
        print_warning "Dry-run mode: using placeholder OpenBLAS paths."
        export MQ_OPENBLAS_ROOT="$ROOTDIR/.dry-run/scipy_openblas32"
        export MQ_OPENBLAS_LIB_DIR="$MQ_OPENBLAS_ROOT/lib"
        if [ "$(uname -s)" == "Darwin" ]; then
            export MQ_OPENBLAS_LIB="$MQ_OPENBLAS_LIB_DIR/libscipy_openblas.dylib"
        else
            export MQ_OPENBLAS_LIB="$MQ_OPENBLAS_LIB_DIR/libscipy_openblas.so"
        fi
        export MQ_OPENBLAS_CMAKE_DIR="$MQ_OPENBLAS_LIB_DIR/cmake/openblas"
        export MQ_OPENBLAS_SYMBOL_PREFIX=scipy_
    elif openblas_info=$(query_openblas_info 2>/dev/null); then
        export_openblas_info "$openblas_info"
    else
        echo "Installing $openblas_python_pkg inside the Python virtual environment"
        call_cmd "$PYTHON" -m pip install --prefer-binary "$openblas_python_pkg"
        openblas_info=$(query_openblas_info) || die "Unable to locate OpenBLAS from scipy-openblas32 after installation."
        export_openblas_info "$openblas_info"
    fi

    if [ "$(uname -s)" == "Darwin" ]; then
        openblas_ld_var=DYLD_LIBRARY_PATH
    else
        openblas_ld_var=LD_LIBRARY_PATH
    fi
    if [ -z "${!openblas_ld_var:-}" ]; then
        export ${openblas_ld_var}="$MQ_OPENBLAS_LIB_DIR"
    else
        export ${openblas_ld_var}="$MQ_OPENBLAS_LIB_DIR:${!openblas_ld_var}"
    fi

    echo "Using OpenBLAS from Python package:"
    echo "  root       = $MQ_OPENBLAS_ROOT"
    echo "  lib dir    = $MQ_OPENBLAS_LIB_DIR"
    echo "  library    = $MQ_OPENBLAS_LIB"
    echo "  cmake dir  = $MQ_OPENBLAS_CMAKE_DIR"
}

# ==============================================================================

if [[ "${created_venv:-0}" -eq 1 || "${do_update_venv:-0}" -eq 1 ]]; then
    critical_pkgs=(pip 'build==0.9.0')

    read_build_system_requires "$ROOTDIR/pyproject.toml"
    critical_pkgs+=("${build_requires[@]}")

    echo "Updating critical Python packages: $PYTHON -m pip install -U ${critical_pkgs[*]}"
    call_cmd "$PYTHON" -m pip install -U "${critical_pkgs[@]}"

    pkgs=(pybind11 "$openblas_python_pkg")

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pkgs+=(auditwheel)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        pkgs+=("delocate==0.10.7")
    fi

    if [ "${cmake_from_venv:-0}" -eq 1 ]; then
        pkgs+=(cmake)
    fi

    if [ "${ninja_from_venv:-0}" -eq 1 ]; then
        pkgs+=(ninja)
    fi

    if [ "${enable_tests:-0}" -eq 1 ]; then
        if [[ -n "$VENV_PYTHON_TEST_PKGS" ]]; then
            read -ra test_pkgs <<< "$VENV_PYTHON_TEST_PKGS"
            pkgs+=( "${test_pkgs[@]}" )
        elif [ "${only_install_pytest:-0}" -eq 1 ]; then
            pkgs+=( pytest pytest-cov pytest-mock mock )
        else
            tmp_file=$(mktemp req_mq_XXX)

            pushd "$ROOTDIR" > /dev/null || exit 1
            "$PYTHON" setup.py gen_reqfile --include-extras=test --output "$tmp_file"
            popd > /dev/null || exit 1

            if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
                while IFS= read -r line || [ -n "$line" ]; do
                    if [ -n "$line" ]; then
                        pkgs+=("$line")
                    fi
                done < "$tmp_file"
            else
                mapfile -t -O "${#pkgs[@]}" pkgs <<< "$(grep '\S' "$tmp_file")"
            fi
            rm -f "$tmp_file"
        fi
    fi

    if [ "${do_docs:-0}" -eq 1 ]; then
        pkgs+=(breathe sphinx sphinx_rtd_theme importlib-metadata myst-parser)
    fi

    # shellcheck disable=SC2154
    if [ -n "${python_extra_pkgs[*]}" ]; then
        pkgs+=("${python_extra_pkgs[@]}")
    fi

    pip_args=(--prefer-binary)
    if [[ ${do_update_venv:-0} -eq 1 ]]; then
        pip_args+=( -U )
    fi

    echo "Updating Python packages: $PYTHON -m pip install ${pip_args[*]} ${pkgs[*]}"
    call_cmd "$PYTHON" -m pip install "${pip_args[@]}" "${pkgs[@]}"
fi

# ==============================================================================

configure_openblas

# ==============================================================================
