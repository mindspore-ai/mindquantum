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

if ($null -eq $_sourced_python_virtualenv_update) { $_sourced_python_virtualenv_update=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ------------------------------------------------------------------------------

. (Join-Path $BASEPATH 'default_values.ps1')
. (Join-Path $BASEPATH 'common_functions.ps1')
. (Join-Path -Path $BASEPATH -ChildPath '..' | Join-Path -ChildPath 'parse_toml.ps1')

# ==============================================================================

if ($null -eq $PYTHON) {
    die '(internal error): PYTHON variable not defined!'
}

if ($null -eq $ROOTDIR) {
    die '(internal error): ROOTDIR variable not defined!'
}

# ==============================================================================

$openblas_python_pkg = 'scipy-openblas32==0.3.31.22.1'

function Query-OpenBlasInfo {
    &"$PYTHON" (Join-Path $ROOTDIR 'scripts\build\locate_openblas.py')
}

function Configure-OpenBlas {
    if ($dry_run) {
        Write-Warning "Dry-run mode: using placeholder OpenBLAS paths."
        $openblas_root = Join-Path $ROOTDIR 'build\.dry-run\scipy_openblas32'
        $openblas_lib_dir = Join-Path $openblas_root 'lib'
        if ($IsWindowsEnv) {
            $openblas_lib = Join-Path $openblas_lib_dir 'libscipy_openblas.dll'
        }
        elseif ($IsMacOSEnv) {
            $openblas_lib = Join-Path $openblas_lib_dir 'libscipy_openblas.dylib'
        }
        else {
            $openblas_lib = Join-Path $openblas_lib_dir 'libscipy_openblas.so'
        }
        $openblas_cmake_dir = Join-Path $openblas_lib_dir 'cmake\openblas'

        $Env:MQ_OPENBLAS_ROOT = "$openblas_root"
        $Env:MQ_OPENBLAS_LIB_DIR = "$openblas_lib_dir"
        $Env:MQ_OPENBLAS_LIB = "$openblas_lib"
        $Env:MQ_OPENBLAS_CMAKE_DIR = "$openblas_cmake_dir"
        $Env:MQ_OPENBLAS_SYMBOL_PREFIX = 'scipy_'
    }
    else {
        $openblas_info = @(Query-OpenBlasInfo 2>$null)
        if ($LastExitCode -ne 0 -or $openblas_info.Count -eq 0) {
            Write-Output "Installing $openblas_python_pkg inside the Python virtual environment"
            Call-Cmd "$PYTHON" -m pip install --prefer-binary "$openblas_python_pkg"
            $openblas_info = @(Query-OpenBlasInfo)
            if ($LastExitCode -ne 0 -or $openblas_info.Count -eq 0) {
                die "Unable to locate OpenBLAS from scipy-openblas32 after installation."
            }
        }

        foreach ($line in $openblas_info) {
            if ($line -match '^([^=]+)=(.*)$') {
                Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
            }
        }

        if (-Not (Test-Path -Path "$Env:MQ_OPENBLAS_LIB")) {
            die "OpenBLAS library file does not exist: $Env:MQ_OPENBLAS_LIB"
        }
    }

    if ($IsWindowsEnv) {
        if ([string]::IsNullOrEmpty($Env:PATH)) {
            $Env:PATH = "$Env:MQ_OPENBLAS_LIB_DIR"
        }
        else {
            $Env:PATH = "$Env:MQ_OPENBLAS_LIB_DIR;$Env:PATH"
        }
        if ([string]::IsNullOrEmpty($Env:MQ_LIB)) {
            $Env:MQ_LIB = "$Env:MQ_OPENBLAS_LIB_DIR"
        }
        else {
            $Env:MQ_LIB = "$Env:MQ_OPENBLAS_LIB_DIR;$Env:MQ_LIB"
        }
    }
    elseif ($IsMacOSEnv) {
        if ([string]::IsNullOrEmpty($Env:DYLD_LIBRARY_PATH)) {
            $Env:DYLD_LIBRARY_PATH = "$Env:MQ_OPENBLAS_LIB_DIR"
        }
        else {
            $Env:DYLD_LIBRARY_PATH = "$Env:MQ_OPENBLAS_LIB_DIR:$Env:DYLD_LIBRARY_PATH"
        }
    }
    else {
        if ([string]::IsNullOrEmpty($Env:LD_LIBRARY_PATH)) {
            $Env:LD_LIBRARY_PATH = "$Env:MQ_OPENBLAS_LIB_DIR"
        }
        else {
            $Env:LD_LIBRARY_PATH = "$Env:MQ_OPENBLAS_LIB_DIR:$Env:LD_LIBRARY_PATH"
        }
    }

    Write-Output 'Using OpenBLAS from Python package:'
    Write-Output "  root       = $Env:MQ_OPENBLAS_ROOT"
    Write-Output "  lib dir    = $Env:MQ_OPENBLAS_LIB_DIR"
    Write-Output "  library    = $Env:MQ_OPENBLAS_LIB"
    Write-Output "  cmake dir  = $Env:MQ_OPENBLAS_CMAKE_DIR"
}

# ==============================================================================

if ($created_venv -or $do_update_venv) {
    Write-Output "Upgrading pip: $PYTHON -m pip install -U pip"
    Call-Cmd "$PYTHON" -m pip install -U pip

    $critical_pkgs = @()

    $build_requires = Read-BuildSystemRequires "$ROOTDIR/pyproject.toml"
    $critical_pkgs += $build_requires

    Write-Output ("Updating critical Python packages: $PYTHON -m pip install -U " + ($critical_pkgs -Join ' '))
    Call-Cmd "$PYTHON" -m pip install -U @critical_pkgs

    $pkgs = @('pybind11', "$openblas_python_pkg")

    if ($IsLinuxEnv) {
        $pkgs += 'auditwheel'
    }
    elseif ($IsMacOSEnv) {
        $pkgs += 'delocate'
    }

    if ($cmake_from_venv) {
        $pkgs += 'cmake'
    }

    if ($ninja_from_venv) {
        $pkgs += 'ninja'
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

            Push-Location "$ROOTDIR"
            Invoke-Expression -Command "$PYTHON setup.py gen_reqfile --include-extras=test --output `"$tmp_file`""
            Pop-Location


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

Configure-OpenBlas

# ==============================================================================
