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

if ($_sourced_python_virtualenv_activate -eq $null) { $_sourced_python_virtualenv_activate=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ------------------------------------------------------------------------------

. (Join-Path $BASEPATH 'default_values.ps1')
. (Join-Path $BASEPATH 'common_functions.ps1')

# ==============================================================================

if ($ROOTDIR -eq $null) {
    die '(internal error): ROOTDIR variable not defined!'
}
if ($PYTHON -eq $null) {
    die '(internal error): PYTHON variable not defined!'
}
if ($python_venv_path -eq $null) {
    die '(internal error): python_venv_path variable not defined!'
}

# ==============================================================================

if ($do_clean_venv) {
    Write-Output "Deleting virtualenv folder: $python_venv_path"
    Call-Cmd Remove-Item -Force -Recurse "'$python_venv_path'" -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------------------

$venv_args = @( "$python_venv_path" )
if ("$Env:VENV_USE_SYSTEM_PACKAGES" -eq '1') {
    $venv_args += '--system-site-packages'
}

$created_venv = $false
if (-Not (Test-Path -Path "'$python_venv_path'" -PathType Container)) {
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

if (-Not $dry_run) {
    . "$activate_path"
} else {
    Write-Output ". $activate_path"
}

# ==============================================================================
