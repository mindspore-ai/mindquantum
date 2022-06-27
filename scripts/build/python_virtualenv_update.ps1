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

if ($_sourced_python_virtualenv_update -eq $null) { $_sourced_python_virtualenv_update=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ------------------------------------------------------------------------------

. (Join-Path $BASEPATH 'default_values.ps1')
. (Join-Path $BASEPATH 'common_functions.ps1')

# ==============================================================================

if ($PYTHON -eq $null) {
    die '(internal error): PYTHON variable not defined!'
}

if ($ROOTDIR -eq $null) {
    die '(internal error): ROOTDIR variable not defined!'
}

# ==============================================================================

if ($created_venv -or $do_update_venv) {
    $critical_pkgs = @('pip', 'setuptools', 'wheel', 'build')
    Write-Output ("Updating critical Python packages: $PYTHON -m pip install -U " + ($critical_pkgs -Join ' '))
    Call-Cmd "$PYTHON" -m pip install -U @critical_pkgs

    $pkgs = @('pybind11')

    if ($IsLinuxEnv) {
        $pkgs += 'auditwheel'
    }
    elseif ($IsMacOSEnv) {
        $pkgs += 'delocate'
    }

    if ($cmake_from_venv) {
        $pkgs += 'cmake'
    }

    if ($enable_tests) {
        if ("$Env:VENV_PYTHON_TEST_PKGS" -ne "") {
            $pkgs += ( "$Env:VENV_PYTHON_TEST_PKGS" -Split ' ' )
        }
        elseif ($only_install_pytest) {
            $pkgs += 'pytest', 'pytest-cov', 'pytest-mock', 'mock'
        }
        else {
            $tmp_file = (New-TemporaryFile).Name

            pushd "$ROOTDIR"
            Invoke-Expression -Command "$PYTHON setup.py gen_reqfile --include-extras=test --output `"$tmp_file`""
            popd

            $tmp = Get-Content -Path "$tmp_file" | Select-String -Pattern '^\s*$' -NotMatch
            $pkgs += $tmp -Split '\n'
            Remove-Item -Force "$tmp_file" -ErrorAction SilentlyContinue
        }
    }

    if ($do_docs) {
        $pkgs += 'breathe', 'sphinx', 'sphinx_rtd_theme', 'importlib-metadata', 'myst-parser'
    }

    if (-Not $python_extra_pkgs -eq $null) {
        $pkgs += $python_extra_pkgs
    }

    $pip_args = @( '--prefer-binary' )
    if ($do_update_venv) {
        $pip_args += '-U'
    }

    # TODO(dnguyen): add wheel delocation package for Windows once we figure this out

    Write-Output ("Updating Python packages: $PYTHON -m pip install " + ($pip_args -Join ' ')  + ($pkgs -Join ' '))
    Call-Cmd "$PYTHON" -m pip install @pip_args @pkgs
}

# ==============================================================================
