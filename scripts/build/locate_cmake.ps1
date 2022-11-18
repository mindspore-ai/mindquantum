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

if ($null -eq $_sourced_locate_cmake) { $_sourced_locate_cmake=1 } else { return }

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

$has_cmake = $false
$cmake_from_venv = $false

foreach($_cmake in @("$python_venv_path\Scripts\cmake",
                     "$python_venv_path\Scripts\cmake.exe",
                     "$python_venv_path\bin\cmake",
                     "$python_venv_path\bin\cmake.exe")) {
    if(Test-Path -Path "$_cmake") {
        $CMAKE = "$_cmake"
        $has_cmake = $true
        $cmake_from_venv = $true
        break
    }
}

# ==============================================================================

$cmake_minimum_str = Get-Content -TotalCount 40 -Path "$ROOTDIR\CMakeLists.txt"
if ("$cmake_minimum_str" -Match "cmake_minimum_required\(VERSION\s+([0-9\.]+)\)") {
    $cmake_version_min = $Matches[1]
}
else {
    $cmake_version_min = "3.20"
}

if(-Not $has_cmake) {
    if(Test-CommandExists cmake3) {
        $CMAKE = "cmake3"
    }
    elseif (Test-CommandExists cmake) {
        $CMAKE = "cmake"
    }

    if ([bool]"$CMAKE") {
        $cmake_version_str = Invoke-Expression -Command "$CMAKE --version"
        if ("$cmake_version_str" -Match "cmake version ([0-9\.]+)") {
            $cmake_version = $Matches[1]
        }

        if ([bool]"$cmake_version" -And [bool]"$cmake_version_min" `
          -And ([System.Version]"$cmake_version_min" -lt [System.Version]"$cmake_version")) {
              $has_cmake=1
          }
    }
}

if (-Not $has_cmake) {
    $pip_args = @('--prefer-binary')
    if ($_IS_MINDSPORE_CI) {
        $pip_args += '-i', 'https://mirror.baidu.com/pypi/simple'
    }

    Write-Output "Installing CMake inside the Python virtual environment"
    Call-Cmd "$PYTHON" -m pip install @pip_args "cmake>=$cmake_version_min"
    foreach($_cmake in @("$python_venv_path\Scripts\cmake",
                         "$python_venv_path\Scripts\cmake.exe",
                         "$python_venv_path\bin\cmake",
                         "$python_venv_path\bin\cmake.exe")) {
        if(Test-Path -Path "$_cmake") {
            $CMAKE = "$_cmake"
            $has_cmake = $true
            $cmake_from_venv = $true
            break
        }
    }
}

# ==============================================================================

Clear-Variable has_cmake
