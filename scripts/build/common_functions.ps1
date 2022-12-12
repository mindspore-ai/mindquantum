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

if ($null -eq $_sourced_common_functions) { $_sourced_common_functions=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ==============================================================================

. (Join-Path $BASEPATH '..\parse_ini.ps1')

# ==============================================================================

function ConvertStringToBool ([string]$value) {
    $value = $value.ToLower();

    switch ($value) {
        '$true' { return $true; }
        "true" { return $true; }
        "1" { return $true; }
        '$false' { return $false; }
        "false" { return $false; }
        "0" { return $false; }
    }
}

# ==============================================================================

function Assign-Value([switch]$Script, [switch]$OnlyOutput) {
    $name = $args[0]
    $value = $args[1]

    if ($Script) {
        $eval_str = "`$script:$name = "
    }
    else {
        $eval_str = "`$$name = "
    }

    if ($value -is [string]) {
        $eval_str += "`"$value`""
    }
    elseif ($value -is [array]) {
        $eval_str += "@({0})" -f (($value | ForEach-Object {"`"$_`""}) -Join ",")
    }
    elseif ($value -is [bool]) {
        if ($value) {
            $eval_str += "`$true"
        }
        else {
            $eval_str += "`$false"
        }
    }
    else {
        $eval_str += "$value"
    }

    if ($OnlyOutput) {
        return $eval_str
    }
    else {
        Write-Debug "$eval_str"
        Invoke-Expression -Command "$eval_str"
    }
}

function Set-Value {
    $name = $args[0]
    if ($args.Length -gt 1) {
        $value = $args[1]
    }
    else {
        $value = $true
    }
    Assign-Value -Script $name $value
    Assign-Value -Script "_${name}_was_set" $true
}

# ------------------------------------------------------------------------------

function Set-VariableFromIni([string]$Path,
                             [string]$TargetSection,
                             [switch]$CheckNull,
                             [switch]$CheckSet,
                             [switch]$DryRun)
{
    $ini_values = Parse-IniFile -Path "$Path"
    if ($TargetSection -ne "") {
        ($ini_values.GetEnumerator() | Where-Object {$_.Name -ne $TargetSection}) | `
          ForEach-Object {$ini_values.Remove($_.Name)}
    }

    # NB: right now we are ignoring the section names... but that might change at some point
    foreach ($section in $ini_values.GetEnumerator()) {
        foreach ($section_value in $section.Value.GetEnumerator()) {
            $name = $section_value.Name.Replace('.', '_')
            $value = $section_value.Value

            if ($section.Name -Match '^.*(path|paths)$') {
                if (-Not [System.IO.Path]::IsPathRooted($value) -And [bool]$value) {
                    $value = Join-Path $ROOTDIR $value
                }
            }

            $eval_str = Assign-Value -OnlyOutput -Script $name $value
            Write-Debug ("{0}  # [{1}]" -f $eval_str.PadRight(50, ' '), $section.Name)

            if($CheckSet) {
                $eval_str = "if (`$_${name}_was_set -eq `$null) { $eval_str }"
            }
            elseif ($CheckNull) {
                $eval_str = "if (`$${name} -eq `$null) { $eval_str }"
            }

            if (-Not $DryRun) {
                # Write-Debug "  invoked expression: $eval_str"
                Invoke-Expression -Command "$eval_str"
            }
        }
    }

}

# ==============================================================================

function die {
    Write-Error "$args"
    Pop-AllEnvironmentVariables
    exit 2
}

# ------------------------------------------------------------------------------

function Call-Cmd {
    if ($dry_run -ne 1) {
        Invoke-Expression -Command "$args"
    }
    else {
        Write-Output "$args"
    }

    if ($LastExitCode -ne 0) {
        Pop-AllEnvironmentVariables
        exit $LastExitCode
    }
}

# ------------------------------------------------------------------------------

function Call-CMake {
    if ($dry_run -ne 1) {
        Write-Output "**********"
        Write-Output "Calling CMake with: cmake $args"
        Write-Output "**********"
    }
    Call-Cmd $CMAKE @args
}

# ==============================================================================

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

# ==============================================================================

$global:_env_var_stack = [System.Collections.ArrayList]@()

function Push-EnvironmentVariables {
    $global:_env_var_stack.Insert(0, @{})
}

function Pop-EnvironmentVariables {
    if (-Not [bool]$global:_env_var_stack) {
        return
    }

    foreach ($h in $global:_env_var_stack[0].GetEnumerator() )
    {
        Write-Debug ("Resetting {0} to '{1}'" -f $h.Name, $h.Value)
        [System.Environment]::SetEnvironmentVariable($h.Name, $h.Value,[System.EnvironmentVariableTarget]::Process)
    }

    $global:_env_var_stack.RemoveAt(0)
}

function Pop-AllEnvironmentVariables {
    while ([bool]$global:_env_var_stack) {
        Pop-EnvironmentVariables
    }
}

function Push-EnvironmentVariable([string]$name, [string]$value) {
    $old_value = [System.Environment]::GetEnvironmentVariable($name, [System.EnvironmentVariableTarget]::Process)
    $global:_env_var_stack[0][$name] = $old_value
    Write-Debug "Locally modifying Env:${name}: replacing old value '$old_value' with '$value'"
    [System.Environment]::SetEnvironmentVariable($name, $value,[System.EnvironmentVariableTarget]::Process)
}

# ==============================================================================

function Protect-ExecString([string]$string) {
    # NB: This will actually work with mis-formatted strings like "AAAA ; BBB'
    return $string -replace "(?!\B('|`")[^`"']*);(?![^`"']*(?:`"|')\B)", '`;'
}

function Convert-StringToArgList([Parameter(ValueFromPipeline=$true)][string]$commandline) {
    $var = Invoke-Expression ("Write-Output {0}" -f (Protect-ExecString $commandline))
    return $var
}
