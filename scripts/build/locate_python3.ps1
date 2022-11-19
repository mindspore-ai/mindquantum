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

if ($null -eq $_sourced_locate_python3) { $_sourced_locate_python3=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ------------------------------------------------------------------------------

. (Join-Path $BASEPATH 'default_values.ps1')
. (Join-Path $BASEPATH 'common_functions.ps1')

# ==============================================================================

if ($null -eq $PYTHON) {
    if(Test-CommandExists python3) {
        $PYTHON = "python3"
    }
    elseif (Test-CommandExists python) {
        $PYTHON = "python"
    }
    else {
        Write-Output 'Unable to locate python or python3!'
        Pop-AllEnvironmentVariables
        exit 1
    }

    Write-Debug "Using Python from: $((Get-Command "$PYTHON").Source)"
}
elseif ($do_verbose) {
    Write-Debug "Using Python from environment variable: $((Get-Command "$PYTHON").Source)"
}

# ==============================================================================
