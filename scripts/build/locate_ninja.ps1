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

if ($null -eq $_sourced_locate_ninja) { $_sourced_locate_ninja=1 } else { return }

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

$has_ninja = $false
$ninja_from_venv = $false

foreach($_ninja in @("$python_venv_path\Scripts\ninja",
                     "$python_venv_path\Scripts\ninja.exe",
                     "$python_venv_path\bin\ninja",
                     "$python_venv_path\bin\ninja.exe")) {
    if(Test-Path -Path "$_ninja") {
        $ninja_exec = "$_ninja"
        $has_ninja = $true
        $ninja_from_venv = $true
        break
    }
}

# ------------------------------------------------------------------------------

if(-Not $has_ninja) {
    if(Test-CommandExists ninja) {
        $ninja_exec = "ninja"
        $has_ninja = $true
    }
}

# ==============================================================================

if (-Not $has_ninja) {
    $pip_args = @()
    if ($_IS_MINDSPORE_CI) {
        $pip_args += '-i', 'https://mirror.baidu.com/pypi/simple'
    }

    Write-Output "Installing Ninja inside the Python virtual environment"
    Call-Cmd "$PYTHON" -m pip install @pip_args ninja
    foreach($_ninja in @("$python_venv_path\Scripts\ninja",
                         "$python_venv_path\Scripts\ninja.exe",
                         "$python_venv_path\bin\ninja",
                         "$python_venv_path\bin\ninja.exe")) {
        if(Test-Path -Path "$_ninja") {
            $ninja_exec = "$_ninja"
            $has_ninja = $true
            $ninja_from_venv = $true
            break
        }
    }
}

# ==============================================================================

Clear-Variable has_ninja
