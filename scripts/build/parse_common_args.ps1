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

if ($_sourced_parse_common_args -eq $null) { $_sourced_parse_common_args=1 } else { return }

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent

# ==============================================================================

if ($ROOTDIR -eq $null) {
    die '(internal error): ROOTDIR variable not defined!'
}

# ==============================================================================

if( $config_file -eq $null) { $config_file = "$ROOTDIR\build.conf" }

if ($Verbose.IsPresent -Or $Debug.IsPresent) {
    $DebugPreference = 'Continue'
    Assign-Value -Script '_verbose_was_set' $true
}

Write-Output "Reading INI/Unix default configuration"
Set-VariableFromIni -Path (Join-Path $BASEPATH 'default_values.conf') -CheckNull

# Other default values (ie. those not present in the config file)
. (Join-Path $BASEPATH 'default_values.ps1')

. (Join-Path $BASEPATH 'common_functions.ps1')

# ==============================================================================

function Print-Show-Libraries {
    Write-Output 'Known third-party libraries:'
    foreach($lib in $third_party_libraries) {
        Write-Output (" - {0}" -f $lib)
    }
}

# ------------------------------------------------------------------------------

function Help-Message() {
    Help-Header

    Write-Output 'Usage:'
    Write-Output ('  {0} [options]' -f $BASENAME)
    Write-Output ''
    Write-Output 'Options:'
    Write-Output '  -H,-Help            Show this help message and exit'
    Write-Output '  -N,$DryRun          Dry run; only print commands but do not execute them'
    Write-Output ''
    Write-Output '  -B,-Build [dir]     Specify build directory'
    Write-Output ("                      Defaults to: {0}" -f $build_dir)
    Write-Output '  -CCache             If ccache or sccache are found within the PATH, use them with CMake'
    Write-Output '  -Clean3rdParty      Clean 3rd party installation directory'
    Write-Output '  -CleanAll           Clean everything before building.'
    Write-Output '                      Equivalent to -CleanVenv -CleanBuildDir'
    Write-Output '  -CleanBuildDir      Delete build directory before building'
    Write-Output '  -CleanCache         Re-run CMake with a clean CMake cache'
    Write-Output '  -CleanVenv          Delete Python virtualenv before building'
    Write-Output '  -Config [dir]       Path to INI configuration file with default values for the parameters'
    Write-Output ("                      Defaults to: {0}" -f $config_file)
    Write-Output '                      NB: command line arguments always take precedence over configuration file values'
    Write-Output '  -Cxx                (experimental) Enable MindQuantum C++ support'
    Write-Output '  -Debug              Build in debug mode'
    Write-Output '  -DebugCMake         Enable debugging mode for CMake configuration step'
    Write-Output '  -Gitee              Use Gitee (where possible) instead of Github/Gitlab'
    Write-Output '  -Gpu                Enable GPU support'
    Write-Output '  -J,-Jobs [N]        Number of parallel jobs for building'
    Write-Output ("                      Defaults to: {0}" -f $n_jobs_default)
    Write-Output '  -LocalPkgs          Compile third-party dependencies locally'
    Write-Output '  -NoConfig           Ignore any configuration file'
    Write-Output '  -Ninja              Build using Ninja instead of make'
    Write-Output '  -Quiet              Disable verbose build rules'
    Write-Output '  -ShowLibraries      Show all known third-party libraries'
    Write-Output '  -V,-Verbose         Enable verbose output from the Bash scripts'
    Write-Output '  -Venv <path>        Path to Python virtual environment'
    Write-Output ("                      Defaults to: {0}" -f $python_venv_path)
    Write-Output '  -With<library>      Build the third-party <library> from source (<library> is case-insensitive)'
    Write-Output '                      (ignored if --local-pkgs is passed, except for projectq)'
    Write-Output '  -Without<library>   Do not build the third-party library from source (<library> is case-insensitive)'
    Write-Output '                      (ignored if --local-pkgs is passed, except for projectq)'
    Write-Output ''
    Write-Output 'Test related options'
    Write-Output '  -Test               Build C++ tests and install dependencies for Python testing as well'
    Write-Output '  -OnlyPytest         Only install pytest and its dependencies when creating/building the virtualenv'
    Write-Output ''
    Write-Output 'CUDA related options:'
    Write-Output '  -CudaArch <arch>    Comma-separated list of architectures to generate device code for.'
    Write-Output '                      Only useful if -Gpu is passed. See CMAKE_CUDA_ARCHITECTURES for more information.'
    Write-Output ''
    Write-Output 'Python related options:'
    Write-Output '  -UpdateVenv         Update the python virtual environment'
    Write-Output ''
    Write-Output 'Developer options:'
    Write-Output '  -CMakeNoRegistry    Disable the use of CMake package registries during configuration'
    Write-Output ''

    if (Test-CommandExists Extra-Help) {
        Extra-Help
    }
}

# ==============================================================================

if ($Help.IsPresent) {
    Help-Message
    exit 1
}

if($ShowLibraries.IsPresent) {
    Print-Show-Libraries
    exit 0
}

# ==============================================================================

if ($CMakeNoRegistry.IsPresent) {
    Set-Value 'cmake_no_registry'
}

if ($DryRun.IsPresent) {
    Set-Value 'dry_run'
}

if ($CCache.IsPresent) {
    Set-Value 'enable_ccache'
}

if ($Clean3rdParty.IsPresent) {
    Set-Value 'do_clean_3rdparty'
}
if ($CleanAll.IsPresent) {
    Set-Value 'do_clean_venv'
    Set-Value 'do_clean_build_dir'
}
if ($CleanBuildDir.IsPresent) {
    Set-Value 'do_clean_build_dir'
}
if ($CleanCache.IsPresent) {
    Set-Value 'do_clean_cache'
}
if ($CleanVenv.IsPresent) {
    Set-Value 'do_clean_venv'
}

if ($Cxx.IsPresent) {
    Set-Value 'enable_cxx'
}

if ($Debug.IsPresent) {
    Set-Value 'build_type' 'Debug'
}

if ($DebugCMake.IsPresent) {
    Set-Value 'cmake_debug_mode'
}

if ($Gitee.IsPresent) {
    Set-Value 'enable_gitee'
}

if ($Gpu.IsPresent) {
    Set-Value 'enable_gpu'
}

if ($LocalPkgs.IsPresent) {
    Set-Value 'force_local_pkgs'
}

if ($Quiet.IsPresent) {
    Set-Value 'cmake_make_silent'
}

if ($Test.IsPresent) {
    Set-Value 'enable_tests'
}

if ($OnlyPytest.IsPresent) {
    Set-Value 'only_install_pytest'
}

if ($UpdateVenv.IsPresent) {
    Set-Value 'do_update_venv'
}

if ([bool]$Build) {
    Set-Value 'build_dir' "$Build"
}

if ([bool]$Config) {
    Set-Value 'config_file' "$Config"
}
if ($NoConfig.IsPresent) {
    Set-Value 'config_file' '__disabled_config__'
}

if ([bool]$CudaArch) {
    Set-Value 'cuda_arch' $CudaArch.Replace(' ', ';').Replace(',', ';')
}

if ($Jobs -ne 0) {
    Set-Value 'n_jobs' $Jobs
}

if ([bool]$Venv) {
    Set-Value 'python_venv_path' "$Venv"
}

if ($Ninja.IsPresent) {
    Set-Value 'cmake_generator' 'Ninja'
}
elseif ($n_jobs -eq -1){
    $n_jobs = $n_jobs_default
}


$unparsed_args = @()

foreach($arg in $args) {
    if ("$arg" -match "[Ww]ith[Oo]ut-?([a-zA-Z0-9_]+)") {
        $enable_lib = $false
        $library = ($Matches[1]).Tolower()
    }
    elseif("$arg" -match "[Ww]ith-?([a-zA-Z0-9_]+)") {
        $enable_lib = $true
        $library = ($Matches[1]).Tolower()
    }
    else {
        $unparsed_args += $arg
        continue
    }

    if (-Not [bool](($third_party_libraries -eq $library) -join " ")) {
        Write-Output ('Unkown library for {0}' -f $arg)
        exit 1
    }

    if ($library -eq "projectq") {
        Set-Value 'enable_projectq' $enable_lib
    }
    elseif ($enable_lib) {
        $local_pkgs += $library
        Assign-Value -Script '_local_pkgs_was_set' $true
    }
    else {
        $local_pkgs = $local_pkgs -ne $library
        Assign-Value -Script '_local_pkgs_was_set' $true
    }
}

$local_pkgs = ($local_pkgs -join ',')

# ==============================================================================

if (Test-Path -Path "$config_file") {
    Write-Output "Reading INI/Unix conf configuration file: $config_file"
    Write-Debug 'NB: overriding values only if not specified on the command line'

    Set-VariableFromIni -Path "$config_file" -CheckSet
}
else {
    Write-Debug "Configuration file ($config_file) ignored because file does not exist"
}

# NB: in case it was set to true in the configuration file
if ($Verbose) {
    $DebugPreference = 'Continue'
}

# ==============================================================================
