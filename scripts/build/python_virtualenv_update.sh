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

if [[ "${created_venv:-0}" -eq 1 || "${do_update_venv:-0}" -eq 1 ]]; then
    critical_pkgs=(pip build)

    read_build_system_requires "$ROOTDIR/pyproject.toml"
    critical_pkgs+=("${build_requires[@]}")

    echo "Updating critical Python packages: $PYTHON -m pip install -U ${critical_pkgs[*]}"
    call_cmd "$PYTHON" -m pip install -U "${critical_pkgs[@]}"

    pkgs=(pybind11)

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pkgs+=(auditwheel)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        pkgs+=(delocate)
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
