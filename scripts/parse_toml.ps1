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

if ($null -eq $_sourced_parse_toml) { $_sourced_parse_toml=1 } else { return }

# ==============================================================================

function Read-BuildSystemRequires([string]$Path) {
    Write-Debug ("Reading 'build-system.requires' from TOML file: {0}" -f $Path)
    $toml_content = ((Get-Content -Raw $Path) -replace 'requires(\s+)=(\s+)', 'requires = ')
    $toml_content = ($toml_content -replace 'requires(\s*)=(\s*)\[', 'requires = [')
    $content = [regex]::Matches(
        $toml_content,
        '(?s)(?<=requires = \[).*?(?=\])'
    ).Value

    if ($null -eq $content) {
        Write-Error ("Failed to parse [build-system.requires] value from {0}" -f $Path)
        exit 1
    }

    $build_requires = $content.Split(' ') `
      | ForEach-Object{$_ -replace "'|\n|\r", '' -replace  ',$', '' -replace '"', "'"} `
      | Where-Object {$_} | ForEach-Object{ "`"$_`""}

    Write-Debug ("  read {0}" -f ($build_requires -Join ' '))

    return $build_requires
}
