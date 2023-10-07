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

[CmdletBinding(PositionalBinding=$false)]

Param(
    [switch]$Analyzer,
    [Alias("B")][ValidateNotNullOrEmpty()][string]$Build,
    [switch]$BuildIsolation,
    [switch]$CCache,
    [switch]$CMakeNoRegistry,
    [switch]$Clean3rdParty,
    [switch]$CleanAll,
    [switch]$CleanBuildDir,
    [switch]$CleanCache,
    [switch]$CleanVenv,
    [ValidateNotNullOrEmpty()][string]$Config,
    [ValidateNotNullOrEmpty()][string]$CudaArch,
    [switch]$Cxx,
    [switch]$DebugCMake,
    [switch]$Delocate,
    [Alias("N")][switch]$DryRun,
    [switch]$FastBuild,
    [ValidateNotNullOrEmpty()][string]$FastBuildDir,
    [switch]$Gitee,
    [switch]$Gpu,
    [Alias("H")][switch]$Help,
    [Alias("J")][ValidateRange(1,10000)][int]$Jobs,
    [switch]$LocalPkgs,
    [switch]$Logging,
    [switch]$Ninja,
    [switch]$NoConfig,
    [switch]$NoBuildIsolation,
    [switch]$NoDelocate,
    [switch]$NoFastBuild,
    [switch]$NoGitee,
    [switch]$OnlyPytest,
    [Alias("O")][ValidateNotNullOrEmpty()][string]$Output,
    [ValidateNotNullOrEmpty()][string]$PlatName,
    [switch]$Quiet,
    [switch]$ShowLibraries,
    [switch]$Test,
    [switch]$UpdateVenv,
    [ValidateNotNullOrEmpty()][string]$Venv,
    [Parameter(Position=1, ValueFromRemainingArguments)]$unparsed_args
)

$BASEPATH = Split-Path $MyInvocation.MyCommand.Path -Parent
$ROOTDIR = $BASEPATH
$PROGRAM = Split-Path $MyInvocation.MyCommand.Path -Leaf
$PARAMETERLIST = (Get-Command -Name ".\$PROGRAM").Parameters

# Test for MindSpore CI
$_IS_MINDSPORE_CI = $false
if ("$Env:JENKINS_URL" -Match 'https?://build.mindspore.cn' -And [bool]$Env:CI -And $Env:CI -eq 1) {
    Write-Output "Detected MindSpore/MindQuantum CI"
    $_IS_MINDSPORE_CI = $true
}
if ([bool]$Env:DEVCLOUD_CI -And $Env:DEVCLOUD_CI -eq 1) {
    Write-Output "Detected MindSpore/MindQuantum CI"
    $_IS_MINDSPORE_CI = $true
}

# ==============================================================================

. (Join-Path $ROOTDIR 'scripts\build\common_functions.ps1')

# ------------------------------------------------------------------------------
# Default values

$python_extra_pkgs = @('wheel-filename>1.2', 'build==0.9.0')

if ($_IS_MINDSPORE_CI ) {
    foreach ($var in @('CUDA_HOME', 'CUDA_PATH')) {
        $value = [Environment]::GetEnvironmentVariable($var)
        if ([bool]$value) {
            Write-Output "$var = $value"
            if (-Not (Test-Path -Path $value)) {
                Write-Warning "$var is set, but location does not exist!"
            }
        }
    }

    $DebugPreference = 'Continue'
    Set-Value 'cmake_debug_mode' $true
    Set-Value 'enable_gitee' $true
    Set-Value 'enable_gpu' $true
}

# ------------------------------------------------------------------------------

function Help-Header {
    Write-Output 'Build binary Python wheel for MindQunantum'
    Write-Output ''
    Write-Output 'This is mainly relevant for developers that want to deploy MindQuantum '
    Write-Output 'on machines other than their own.'
    Write-Output ''
    Write-Output 'This script will create a Python virtualenv in the MindQuantum root'
    Write-Output 'directory and then build a binary Python wheel of MindQuantum.'
}

function Extra-Help {
    Write-Output 'Extra options:'
    Write-Output '  -(No)BuildIsolation  Pass --no-isolation to python3 -m build'
    Write-Output '  -(No)Delocate        Delocate the binary wheels after build is finished'
    Write-Output '                       (enabled by default; pass -NoDelocate to disable)'
    Write-Output '  -(No)FastBuild       If possible use an existing CMake directory to build the C++ Python extensions'
    Write-Output '                       instead of using the normal Python bdist_wheel process. Assumes that '
    Write-Output '                       IN_PLACE_BUILD=ON'
    Write-Output '                       Use this with caution. CI build should not be using this.'
    Write-Output '  -FastBuildDir        Specify build directory when performing a fast-build'
    Write-Output '  -O,-Output [dir]     Output directory for built wheels'
    Write-Output '  -P,-PlatName [dir]   Platform name to use for wheel delocation'
    Write-Output '                       (only effective if -Delocate is used)'
    Write-Output ''
    Write-Output 'Example calls:'
    Write-Output "$PROGRAM"
    Write-Output "$PROGRAM -Gpu"
    Write-Output "$PROGRAM -Cxx -WithBoost -Without-gmp -Venv D:\venv"
}

# ------------------------------------------------------------------------------

. (Join-Path $ROOTDIR 'scripts\build\parse_common_args.ps1') @args @unparsed_args

Write-Debug 'Bound PowerShell parameters'
foreach ($Parameter in $PARAMETERLIST) {
    Get-Variable -Name $Parameter.Values.Name -ErrorAction SilentlyContinue `
      | ForEach-Object { Write-Debug ("{0,-40} {1}" -f $_.Name, $_.Value)}
}


if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ------------------------------------------------------------------------------

if (([bool]$Delocate)) {
    Set-Value 'delocate_wheel'
}

if (([bool]$NoDelocate)) {
    Set-Value 'delocate_wheel' $false
}

if (([bool]$BuildIsolation)) {
    Set-Value 'build_isolation'
}

if (([bool]$NoBuildIsolation)) {
    Set-Value 'build_isolation' $false
}

if (([bool]$FastBuild)) {
    Set-Value 'fast_build'
}

if (([bool]$NoFastBuild)) {
    Set-Value 'fast_build' $false
}

if ([bool]$FastBuildDir) {
    Set-Value 'fast_build_dir' "$FastBuildDir"
}

if ([bool]$Output) {
    Set-Value 'output_path' "$Output"
}

if ([bool]$PlatName) {
    Set-Value 'platform_name' "$PlatName"
}

# ==============================================================================
# Locate python or python3

. (Join-Path $ROOTDIR 'scripts\build\locate_python3.ps1')

if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ==============================================================================

$ErrorActionPreference = 'Stop'

Write-Output "Called with: $($MyInvocation.Line)"

cd "$ROOTDIR"

# ------------------------------------------------------------------------------

# NB: `created_venv` variable can be used to detect if a virtualenv was created or not
. (Join-Path $ROOTDIR 'scripts\build\python_virtualenv_activate.ps1')

if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ------------------------------------------------------------------------------
# Locate cmake or cmake3

# NB: `cmake_from_venv` variable is set by this script (and is used by python_virtualenv_update.sh)
. (Join-Path $ROOTDIR 'scripts\build\locate_cmake.ps1')

if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ------------------------------------------------------------------------------
# Locate ninja if needed

if (([bool]$Ninja)) {
    # NB: `ninja_from_venv` variable is set by this script (and is used by python_virtualenv_update.sh)
    . (Join-Path $ROOTDIR 'scripts\build\locate_ninja.ps1')

    if ($LastExitCode -ne 0) {
        exit $LastExitCode
    }
}

# ------------------------------------------------------------------------------

# Update Python virtualenv (if requested/necessary)

. (Join-Path $ROOTDIR 'scripts\build\python_virtualenv_update.ps1')

if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ------------------------------------------------------------------------------
# Setup arguments for build

if ($_build_dir_was_set -eq $null) { $_build_dir_was_set = $false }

$build_args = @()

$cmake_option_names = @{
    cmake_debug_mode = 'ENABLE_CMAKE_DEBUG'
    do_clean_3rdparty = 'CLEAN_3RDPARTY_INSTALL_DIR'
    enable_analyzer = 'ENABLE_ANALYZER'
    enable_gitee = 'ENABLE_GITEE'
    enable_gpu = 'ENABLE_CUDA'
    enable_logging = 'ENABLE_LOGGING'
    enable_tests = 'BUILD_TESTING'
    logging_enable_debug = 'ENABLE_LOGGING_DEBUG_LEVEL'
    logging_enable_trace = 'ENABLE_LOGGING_TRACE_LEVEL'
}

foreach ($el in $cmake_option_names.GetEnumerator()) {
    $value = Invoke-Expression -Command ("`${0}" -f $el.Name)

    if ($value) {
        $build_args += '--set', "$($el.Value)"
    }
    else {
        $build_args += '--unset', "$($el.Value)"
    }
}

if ($_IS_MINDSPORE_CI) {
    $build_args += '--set', 'MINDSPORE_CI'
}

if ($cmake_make_silent) {
    $build_args += '--unset', 'USE_VERBOSE_MAKEFILE'
}
else {
    $build_args += '--set', 'USE_VERBOSE_MAKEFILE'
}

if ($cmake_no_registry) {
    $build_args += '--unset', 'CMAKE_FIND_USE_PACKAGE_REGISTRY'
    $build_args += '--unset', 'CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY'
}
else {
    $build_args += '--set', 'CMAKE_FIND_USE_PACKAGE_REGISTRY'
    $build_args += '--set', 'CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY'
}

if ([bool]$cmake_generator) {
    $build_args += '-G', "$cmake_generator"
}

if ($fast_build) {
    $build_args += 'bdist_wheel', '--fast-build'

    if ([bool]$fast_build_dir) {
        $build_args += 'bdist_wheel', '--fast-build-dir', "$fast_build_dir"
    }
}

if ($n_jobs -ne -1) {
    $build_args += '--var', 'JOBS', '$n_jobs'
    $build_args += 'build_ext', "--jobs=$n_jobs"
}

if ($build_type -eq 'Debug') {
    $build_args += 'build', '--debug'
}

if ($_build_dir_was_set) {
    $build_args += 'build_ext', '--build-dir', "$build_dir"
}

if ($force_local_pkgs) {
    $build_args += '--var', 'MQ_FORCE_LOCAL_PKGS', 'all'
}
elseif ([bool]"$local_pkgs") {
    $build_args += '--var', 'MQ_FORCE_LOCAL_PKGS', "`"$local_pkgs`""
}

# --------------------------------------

if ($enable_gpu -And [bool]$cuda_arch) {
    Write-Error -Category NotImplemented -Message "-CudaArch is unsupported (thus ignored) with $PROGRAM!"
}

if ($enable_ccache) {
    $ccache_exec=''
    if(Test-CommandExists ccache) {
        $ccache_exec = 'ccache'
    }
    elseif(Test-CommandExists sccache) {
        $ccache_exec = 'sccache'
    }

    if ([bool]$ccache_exec) {
        $ccache_exec = (Get-Command "$ccache_exec").Source
        $build_args += '--var', 'CMAKE_C_COMPILER_LAUNCHER', "$ccache_exec"
        $build_args += '--var', 'CMAKE_CXX_COMPILER_LAUNCHER', "$ccache_exec"
        if([bool]$enable_gpu) {
            $build_args += '--var', 'CMAKE_CUDA_COMPILER_LAUNCHER', "$ccache_exec"
        }
    }
}

# NB: CMake < 3.24 typically set CC, CXX during the first run, which basically overwrites the values in CC, CXX. In
#     order to work around that, we explicitly set the compilers using the related CMake variables.
if([bool]$Env:CC) {
    $build_args += '--var', 'CMAKE_C_COMPILER', "$Env:CC"
}
if([bool]$Env:CXX) {
    $build_args += '--var', 'CMAKE_CXX_COMPILER', "$Env:CXX"
}
if([bool]$Env:CUDACXX) {
    $build_args += '--var', 'CMAKE_CUDA_COMPILER', "$Env:CUDACXX"
}

Write-Debug 'Will be passing these arguments to setup.py:'
Write-Debug "    $build_args"

# ==============================================================================

if ([bool]$enable_gpu) {
    # Older CMake using find_package(CUDA) would rely on CUDA_HOME, but newer CMake only look at CUDACXX and CUDA_PATH
    if ([bool]$Env:CUDA_HOME -And -Not [bool]$Env:CUDA_PATH) {
        Write-Output 'CUDA_HOME is defined, but CUDA_PATH is not. Setting CUDA_PATH=CUDA_HOME'
        [System.Environment]::SetEnvironmentVariable('CUDA_PATH',"$Env:CUDA_HOME",[System.EnvironmentVariableTarget]::Process)
    }
    Write-Debug "CUDA_PATH = $Env:CUDA_PATH"
}

# ==============================================================================

# Convert the CMake arguments for passing them using -C to python3 -m build
$fixed_args = @()
foreach($arg in $build_args) {
    $fixed_args += "-C--global-option=$arg"
}

$unparsed_args = $unparsed_args | Where-Object {$_} | ForEach-Object { "'$_'" }
if ([bool]$unparsed_args) {
    $fixed_args += $unparsed_args
}

$build_args = @('-w')
if (-Not $build_isolation) {
    $build_args += "--no-isolation"
}

# ------------------------------------------------------------------------------
# Build the wheels

if ($_build_dir_was_set) {
    if ($do_clean_build_dir) {
        Write-Output "Deleting build folder: $build_dir"
        Call-Cmd Remove-Item -Force -Recurse "'$build_dir'" -ErrorAction SilentlyContinue
    }
    elseif ($do_clean_cache) {
        Write-Output "Removing CMake cache at: $build_dir/CMakeCache.txt"
        Call-Cmd Remove-Item -Force "'$build_dir/CMakeCache.txt'" -ErrorAction SilentlyContinue
        Write-Output "Removing CMake files at: $build_dir/CMakeFiles"
        Call-Cmd Remove-Item -Force -Recurse "'$build_dir/CMakeFiles'" -ErrorAction SilentlyContinue
        Write-Output "Removing CMake files at: $build_dir/cmake-ldtest*"
        Call-Cmd Remove-Item -Force -Recurse "'$build_dir/cmake-ldtest*'" -ErrorAction SilentlyContinue
    }
}

if ($delocate_wheel) {
    $Env:MQ_DELOCATE_WHEEL = 1

    if ([bool]$platform_name) {
        $Env:MQ_DELOCATE_WHEEL_PLAT = "$platform_name"
    }

    if ([bool]$_build_dir_was_set -Or [bool]$fast_build) {
        $build_dir_for_env = $build_dir
    }
    elseif ([bool]$_fast_build_dir_was_set) {
        $build_dir_for_env = $fast_build_dir
    }
    else {
        $build_dir_for_env = (&"$PYTHON" -m mindquantum_config --tempdir)
    }

    if ($_IS_MINDSPORE_CI) {
        $Env:MQ_LIB_PATHS = "$ROOTDIR/ld_library_paths.txt"
    }
    else {
        $Env:MQ_LIB_PATHS = "$build_dir_for_env/ld_library_paths.txt"
    }
    $Env:MQ_BUILD_DIR = "$build_dir_for_env"

    Write-Debug "MQ_LIB_PATHS = $Env:MQ_LIB_PATHS"
    Write-Debug "MQ_BUILD_DIR = $Env:MQ_BUILD_DIR"
}
else {
    $Env:MQ_DELOCATE_WHEEL = 0
}

Call-Cmd "$PYTHON" -m build @build_args @fixed_args
$Env:MQ_LIB_PATHS = ''
$Env:MQ_BUILD_DIR = ''
if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

# ------------------------------------------------------------------------------

if (Test-Path -Path "$output_path") {
    Call-Cmd Remove-Item -Force -Recurse "'$output_path'" -ErrorAction SilentlyContinue
}

Call-Cmd New-Item -Path "'$output_path'" -ItemType "directory"

Call-Cmd Move-Item -Path "'$ROOTDIR\dist\*.whl'" -Destination "$output_path"

Call-Cmd Write-Output "------Successfully created mindquantum package------"

# ==============================================================================

<#
.SYNOPSIS

Performs monthly data updates.

.DESCRIPTION

Build MindQunantum locally (in-source build)

This is mainly relevant for developers that do not want to always have to reinstall the Python package

This script will create a Python virtualenv in the MindQuantum root directory and then build all the C++ Python
modules and place the generated libraries in their right locations within the MindQuantum folder hierarchy so Python
knows how to find them.

A pth-file will be created in the virtualenv site-packages directory so that the MindQuantum root folder will be added
to the Python PATH without the need to modify PYTHONPATH.

.PARAMETER Analyzer
Use the compiler static analysis tool during compilation (GCC & MSVC)

.PARAMETER Build
Specify build directory. Defaults to: Path\To\Script\build

.PARAMETER BuildIsolation
Do not pass --no-isolation to python3 -m build

.PARAMETER CCache
If ccache or sccache are found within the PATH, use them with CMake

.PARAMETER CMakeNoRegistry
Disable the use of CMake package registries during configuration

.PARAMETER Clean3rdParty
Clean 3rd party installation directory

.PARAMETER CleanAll
Clean everything before building.
Equivalent to -CleanVenv -CleanBuildDir

.PARAMETER CleanBuildDir
Delete build directory before building

.PARAMETER CleanCache
Re-run CMake with a clean CMake cache

.PARAMETER CleanVenv
Delete Python virtualenv before building

.PARAMETER CMakeNoRegistry
Do not use the CMake registry to find packages

.PARAMETER Config
Path to INI configuration file with default values for the parameters

.PARAMETER Debug
Build in debug mode

.PARAMETER DebugCMake
Enable debugging mode for CMake configuration step

.PARAMETER DryRun
Dry run; only print commands but do not execute them

.PARAMETER Delocate
Delocate the binary wheels after build is finished (enabled by default; pass -NoDelocate to disable)

.PARAMETER FastBuild
If possible use an existing CMake directory to build the C++ Python extensions instead of using the normal Python
bdist_wheel process.Use this with caution.
CI build should not be using this.

.PARAMETER FastBuildDir
Specify build directory when performing a fast-build. See help message using -Help.

.PARAMETER Gitee
Use Gitee (where possible) instead of Github/Gitlab

.PARAMETER Gpu
Enable GPU support

.PARAMETER Help
Show help message.

.PARAMETER Jobs
Number of parallel jobs for building

.PARAMETER LocalPkgs
Compile third-party dependencies locally

.PARAMETER Logging
Enable logging in C++ code

.PARAMETER Ninja
Build using Ninja instead of make

.PARAMETER NoBuildIsolation
Pass --no-isolation to python3 -m build

.PARAMETER NoConfig
Ignore any configuration file

.PARAMETER NoDelocate
Do not delocate the binary wheels after build is finished (pass -Delocate to enable)

.PARAMETER NoFastBuild
Do not use a "fast" build process when building a wheel. See doc for -FastBuild.

.PARAMETER NoGitee
Do not favor Gitee over Github/Gitlab

.PARAMETER OnlyPytest
Only install pytest and its dependencies when creating/building the virtualenv

.PARAMETER Output
Output directory for built wheels

.PARAMETER PlatName
Platform name to use for wheel delocation (only effective if -Delocate is used)

.PARAMETER Quiet
Disable verbose build rules

.PARAMETER ShowLibraries
Show all known third-party libraries.

.PARAMETER Test
Build C++ tests and install dependencies for Python testing as well

.PARAMETER Verbose
Enable verbose output from the Bash scripts

.PARAMETER Venv
Path to Python virtual environment. Defaults to: Path\To\Script\venv

.PARAMETER UpdateVenv
Update the python virtual environment

.PARAMETER CudaArch
Comma-separated list of architectures to generate device code for.
Only useful if -Gpu is passed. See CMAKE_CUDA_ARCHITECTURES for more information.

.INPUTS

None.

.OUTPUTS

None.

.EXAMPLE

PS> .\build.ps1

.EXAMPLE

PS> .\build.ps1 -gpu

.EXAMPLE

PS> .\build.ps1 -Cxx -WithBoost -WithoutEigen3

.EXAMPLE

PS> .\build.ps1 -Gpu -DCMAKE_CUDA_COMPILER=D:\cuda\bin\nvcc
#>
