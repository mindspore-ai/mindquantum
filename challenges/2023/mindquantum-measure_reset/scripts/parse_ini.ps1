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

if ($null -eq $_sourced_parse_ini) { $_sourced_parse_ini=1 } else { return }

# ==============================================================================

function Convert-IniValue($value)
{
    $ini = [ordered]@{}
    switch -regex ("$value")
    {
        "^\s*(\d+)\s*$" {
            return [int]$matches[1]
        }
        "^\s*(\d+\.\d+)\s*$" {
            return [decimal]$matches[1]
        }
        "^\s*(true|yes)\s*$" {
            return $true
        }
        "^\s*(false|no)\s*$" {
            return $false
        }
        default {
            return $value.Trim()
        }
    }
}

function Parse-IniFile ([string]$Path)
{
    $ini = [ordered]@{}
    $section = 'main'
    $ini[$section] = [ordered]@{}
    switch -regex -file $Path
    {
        '^(;|#.*)$' { continue }

        #Section.
        '^\[(.+)\]$' {
            $section = $matches[1].Trim()
            $ini[$section] = [ordered]@{}
            continue
        }

        # Array values
        '^\s*([^#][\w\d_-]+?)\[]\s*=\s*(.*)' {
            $name,$value = $matches[1..2]
            if (!$ini[$section][$name]) {
                $ini[$section][$name] = @()
            }
            $value = Convert-IniValue "$value"
            $ini[$section][$name] += $value
            continue
        }

        #Array associated
        '^\s*([^#][\w\d_-]+?)\[([\w\d_-]+?)]\s*=\s*(.*)' {
            $name, $association, $value = $matches[1..3]
            if (!$ini[$section][$name]) {
                $ini[$section][$name] = [ordered]@{}
            }
            $value = Convert-IniValue "$value"
            $ini[$section][$name].Add($association, $value)
            continue
        }

        # Values
        '^\s*([^#][\w\d_-]+?)\s*=\s*(.*)' {
            $name,$value = $matches[1..2]
            $ini[$section][$name] = Convert-IniValue "$value"
            continue
        }
    }

    # Remove empty sections
    ($ini.GetEnumerator() | Where-Object { $_.Value.Count -eq 0 }) | ForEach-Object { $ini.Remove($_.Name) }

    return $ini
}
