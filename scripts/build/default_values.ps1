# Copyright 2021 Huawei Technologies Co., Ltd
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

if ($null -eq $_sourced_default_values) { $_sourced_default_values=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ==============================================================================
# Default values for input arguments

$IsLinuxEnv = [bool](Get-Variable -Name "IsLinux" -ErrorAction Ignore)
$IsMacOSEnv = [bool](Get-Variable -Name "IsMacOS" -ErrorAction Ignore)
$IsWinEnv = !$IsLinuxEnv -and !$IsMacOSEnv

function Test-CommandExists{
    Param ($command)

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Stop'

    try {
        if(Get-Command $command) {
            return $TRUE
        }
        else {
            return $FALSE
        }
    }
    Catch {
        return $FALSE
    }
    Finally {
        $ErrorActionPreference=$oldPreference
    }
}

# ------------------------------------------------------------------------------

if ($null -eq $n_jobs_default) {
    $n_jobs_default = 8
    if(Test-CommandExists nproc) {
        $n_jobs_default = nproc
    }
    elseif (Test-CommandExists sysctl) {
        $n_jobs_default = Invoke-Expression -Command "sysctl -n hw.logicalcpu"
    }
    elseif ($IsWinEnv -eq 1) {
        if (Test-CommandExists Get-CimInstance) {
            $n_jobs_default = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
        }
        elseif (Test-CommandExists wmic) {
            $tmp = (wmic cpu get NumberOfLogicalProcessors /value) -Join ' '
            if ($tmp -match "\s*[a-zA-Z]+=([0-9]+)") {
                $n_jobs_default = $Matches[1]
            }
        }
    }
}

# ==============================================================================

$third_party_libraries = ((Get-ChildItem -Path "$ROOTDIR\third_party" -Directory -Exclude cmake).Name).ForEach("ToLower")

# ==============================================================================
# Other helper variables

if ($null -eq $cmake_from_venv) { $cmake_from_venv = $false }
if ($null -eq $ninja_from_venv) { $ninja_from_venv = $false }
