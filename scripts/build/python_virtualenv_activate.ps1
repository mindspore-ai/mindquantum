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

if ($null -eq $_sourced_python_virtualenv_activate) { $_sourced_python_virtualenv_activate=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ------------------------------------------------------------------------------

. (Join-Path $BASEPATH 'default_values.ps1')
. (Join-Path $BASEPATH 'common_functions.ps1')

# ==============================================================================

if ($null -eq $ROOTDIR) {

    die '(internal error): ROOTDIR variable not defined!'
}
if ($null -eq $PYTHON) {
    die '(internal error): PYTHON variable not defined!'
}
if ($null -eq $python_venv_path) {
    die '(internal error): python_venv_path variable not defined!'
}

# ==============================================================================

if ($do_clean_venv) {
    Write-Output "Deleting virtualenv folder: $python_venv_path"
    Call-Cmd Remove-Item -Force -Recurse "'$python_venv_path'" -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------------------

$venv_args = @( "'$python_venv_path'" )
if ("$Env:VENV_USE_SYSTEM_PACKAGES" -eq '1') {
    $venv_args += '--system-site-packages'
}

$created_venv = $false
if (-Not (Test-Path -Path "$python_venv_path" -PathType Container)) {
    $created_venv = $true
    Write-Output "Creating Python virtualenv: $PYTHON -m venv $venv_args"
    Call-Cmd "$PYTHON" -m venv @venv_args
}
elseif ($do_update_venv) {
    $venv_args += '--upgrade'
    Write-Output "Updating Python virtualenv: $PYTHON -m venv $venv_args"
    Call-Cmd "$PYTHON" -m venv @venv_args
}

if($IsWinEnv) {
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
}

Write-Output "Activating Python virtual environment: $python_venv_path"
$activate_path = Join-Path $python_venv_path 'bin\Activate.ps1'
if (Test-Path -Path (Join-Path $python_venv_path 'Scripts\Activate.ps1') -PathType Leaf) {
    $activate_path = (Join-Path $python_venv_path 'Scripts\Activate.ps1')
}

$activated_venv = $false
if (-Not $dry_run) {
    . $activate_path
    $activated_venv = $true
} else {
    Write-Output ". $activate_path"

    if (Test-Path -Type Leaf -Path $activate_path) {
        . $activate_path
        $activated_venv = $true
    }
}

Write-Debug "  activated_venv = $activated_venv"

# ==============================================================================

if ($activated_venv) {
    $adjust_python = $true

    foreach ($subdir in @('bin', 'Scripts')) {
        foreach ($ext in @('', '.exe')) {
            $python_exec = "${Env:VIRTUAL_ENV}\${subdir}\${PYTHON}${ext}"
            Write-Debug "    trying '$python_exec'"
            if (Test-Path -Path $python_exec) {
                $adjust_python = $false
                break
            }
        }
    }

    if ($adjust_python) {
        Write-Output "$PYTHON not found in $Env:VIRTUAL_ENV"
        Write-Output "  -> looking for Python executables in $ENV:VIRTUAL_ENV"

        $found = $false
        foreach ($subdir in @('bin', 'Scripts')) {
            foreach ($exec in @('python3', 'python', 'python3.exe', 'python.exe')) {
                $python_exec = "${Env:VIRTUAL_ENV}\${subdir}\${exec}"
                Write-Debug "    trying '$python_exec'"
                if (Test-Path -Path $python_exec) {
                    $PYTHON = $exec
                    $found = $true
                    break
                }
            }
        }

        if (-Not $found) {
            Write-Error 'Unable to locate python or python3 in virtual environment!'
            Pop-AllEnvironmentVariables
            exit 1
        }

        Write-Output "  -> using $PYTHON"
    }
}

# ==============================================================================
