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
    [switch]$CCache,
    [switch]$CMakeNoRegistry,
    [switch]$Clean,
    [switch]$Clean3rdParty,
    [switch]$CleanAll,
    [switch]$CleanBuildDir,
    [switch]$CleanCache,
    [switch]$CleanVenv,
    [ValidateNotNullOrEmpty()][string]$Config,
    [Alias("C")][switch]$Configure,
    [switch]$ConfigureOnly,
    [ValidateNotNullOrEmpty()][string]$CudaArch,
    [switch]$Cxx,
    [switch]$DebugCMake,
    [Alias("N")][switch]$DryRun,
    [Alias("Docs")][switch]$Doc,
    [switch]$Gitee,
    [switch]$Gpu,
    [Alias("H")][switch]$Help,
    [switch]$Install,
    [Alias("J")][ValidateRange(1,100000)][int]$Jobs,
    [switch]$LocalPkgs,
    [switch]$Logging,
    [switch]$LoggingDebug,
    [switch]$LoggingTrace,
    [switch]$Ninja,
    [switch]$NoConfig,
    [switch]$NoGitee,
    [switch]$OnlyPytest,
    [ValidateNotNullOrEmpty()][string]$Prefix,
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
$_IS_MINDSPORE_CI=$false
if ("$Env:JENKINS_URL" -Match 'https?://build.mindspore.cn' -And [bool]$Env:CI) {
    Write-Output "Detected MindSpore/MindQuantum CI"
    $_IS_MINDSPORE_CI=$true
}

# ==============================================================================
# Default values

. (Join-Path $ROOTDIR 'scripts\build\common_functions.ps1')

Push-EnvironmentVariables

# ------------------------------------------------------------------------------

function Help-Header {
    Write-Output 'Build MindQunantum locally (in-source build)'
    Write-Output ''
    Write-Output 'This is mainly relevant for developers that do not want to always '
    Write-Output 'have to reinstall the Python package'
    Write-Output ''
    Write-Output 'This script will create a Python virtualenv in the MindQuantum root'
    Write-Output 'directory and then build all the C++ Python modules and place the'
    Write-Output 'generated libraries in their right locations within the MindQuantum'
    Write-Output 'folder hierarchy so Python knows how to find them.'
    Write-Output ''
    Write-Output 'A pth-file will be created in the virtualenv site-packages directory'
    Write-Output 'so that the MindQuantum root folder will be added to the Python PATH'
    Write-Output 'without the need to modify PYTHONPATH.'
}

function Extra-Help {
    Write-Output 'Extra options:'
    Write-Output '  -CCache             If ccache or sccache are found within the PATH, use them with CMake'
    Write-Output '  -Clean              Run make clean before building'
    Write-Output '  -C,-Configure       Force running the CMake configure step'
    Write-Output '  -ConfigureOnly      Stop after the CMake configure and generation steps (ie. before building MindQuantum)'
    Write-Output '  -Doc, -Docs         Setup the Python virtualenv for building the documentation and ask CMake to build the'
    Write-Output '                      documentation'
    Write-Output '  -Install            Build the ´install´ target'
    Write-Output '  -Prefix             Specify installation prefix'
    Write-Output ''
    Write-Output 'Any options not matching one of the above will be passed on to CMake during the configuration step. In addition, any'
    Wirte-Output 'options after "--%" will be passed onto CMake during the configuration step'
    Write-Output ''
    Write-Output 'Example calls:'
    Write-Output ("{0} -B build" -f $PROGRAM)
    Write-Output ("{0} -B build -Gpu" -f $PROGRAM)
    Write-Output ("{0} -B build -Cxx -WithBoost -Without-Eigen3" -f $PROGRAM)
    Write-Output ("{0} -B build -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc" -f $PROGRAM)
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

if (([bool]$Clean)) {
    Set-Value 'do_clean'
}

if (([bool]$C) -or ([bool]$Configure)) {
    Set-Value 'do_configure'
}
if (([bool]$ConfigureOnly)) {
    Set-Value 'configure_only'
}

if (([bool]$Doc)) {
    Set-Value 'do_docs'
}

if (([bool]$Install)) {
    Set-Value 'do_install'
}

if ([bool]$Prefix) {
    Set-Value 'prefix_dir' "$Prefix"
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
# Create a virtual environment for building the wheel

if ($do_clean_build_dir) {
    Write-Output "Deleting build folder: $build_dir"
    Call-Cmd Remove-Item -Force -Recurse "'$build_dir'" -ErrorAction SilentlyContinue
}

# NB: `created_venv` variable can be used to detect if a virtualenv was created or not
. (Join-Path $ROOTDIR 'scripts\build\python_virtualenv_activate.ps1')

if ($LastExitCode -ne 0) {
    exit $LastExitCode
}

if ($dry_run -ne 1) {
    # Make sure the root directory is in the virtualenv PATH
    $site_pkg_dir = Invoke-Expression -Command "$PYTHON -c 'import site; print(site.getsitepackages()[0])'"
    $pth_file = "$site_pkg_dir\mindquantum_local.pth"

    if (-Not (Test-Path -Path "$pth_file" -PathType leaf)) {
        Write-Output "Creating pth-file in $pth_file"
        Write-Output "$ROOTDIR" > "$pth_file"
    }
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

$CMAKE_BOOL = @('OFF', 'ON')

$cmake_args = @('-DIN_PLACE_BUILD:BOOL=ON'
                '-DIS_PYTHON_BUILD:BOOL=OFF'
                '-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON'
                "-DCMAKE_BUILD_TYPE:STRING={0}" -f $build_type
                "-DENABLE_ANALYZER:BOOL={0}" -f $CMAKE_BOOL[$enable_analyzer]
                "-DENABLE_CMAKE_DEBUG:BOOL={0}" -f $CMAKE_BOOL[$cmake_debug_mode]
                "-DENABLE_CUDA:BOOL={0}" -f $CMAKE_BOOL[$enable_gpu]
                "-DENABLE_DOCUMENTATION:BOOL={0}" -f $CMAKE_BOOL[$do_docs]
                "-DENABLE_GITEE:BOOL={0}" -f $CMAKE_BOOL[$enable_gitee]
                "-DENABLE_LOGGING:BOOL={0}" -f $CMAKE_BOOL[$enable_logging]
                "-DENABLE_LOGGING_DEBUG_LEVEL:BOOL={0}" -f $CMAKE_BOOL[$logging_enable_debug]
                "-DENABLE_LOGGING_TRACE_LEVEL:BOOL={0}" -f $CMAKE_BOOL[$logging_enable_trace]
                "-DBUILD_TESTING:BOOL={0}" -f $CMAKE_BOOL[$enable_tests]
                "-DCLEAN_3RDPARTY_INSTALL_DIR:BOOL={0}" -f $CMAKE_BOOL[$do_clean_3rdparty]
                "-DUSE_VERBOSE_MAKEFILE:BOOL={0}" -f $CMAKE_BOOL[-not $cmake_make_silent]
                "-DCMAKE_FIND_USE_PACKAGE_REGISTRY:BOOL={0}" -f $CMAKE_BOOL[-not $cmake_no_registry]
                "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY:BOOL:BOOL={0}" -f $CMAKE_BOOL[-not $cmake_no_registry]
               )
$make_args = @()

if ([bool]$cmake_generator) {
    $cmake_args += "-G", "'$cmake_generator'"
}

if([bool]$prefix_dir) {
    $cmake_args += "-DCMAKE_INSTALL_PREFIX:FILEPATH=`"${prefix_dir}`""
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
        $cmake_args += "-DCMAKE_C_COMPILER_LAUNCHER=`"$ccache_exec`""
        $cmake_args += "-DCMAKE_CXX_COMPILER_LAUNCHER=`"$ccache_exec`""
        if([bool]$enable_gpu) {
            $cmake_args += "-DCMAKE_CUDA_COMPILER_LAUNCHER=`"$ccache_exec`""
        }
    }
}

if ($enable_gpu -and [bool]$cuda_arch) {
    $cmake_args += "-DCMAKE_CUDA_ARCHITECTURES:STRING=`"$cuda_arch`""
}

if ($force_local_pkgs) {
    $cmake_args += "-DMQ_FORCE_LOCAL_PKGS=all"
}
elseif ([bool]"$local_pkgs") {
    $cmake_args += "-DMQ_FORCE_LOCAL_PKGS=`"$local_pkgs`""
}

if($n_jobs -ne -1) {
    $cmake_args += "-DJOBS:STRING={0}" -f $n_jobs
    $make_args += "-j `"$n_jobs`""
}

# NB: CMake < 3.24 typically set CC, CXX during the first run, which basically overwrites the values in CC, CXX. In
#     order to work around that, we explicitly set the compilers using the related CMake variables.
if([bool]$Env:CC) {
    $cmake_args += "-DCMAKE_C_COMPILER:FILEPATH=$Env:CC"
}
if([bool]$Env:CXX) {
    $cmake_args += "-DCMAKE_CXX_COMPILER:FILEPATH=$Env:CXX"
}
if([bool]$Env:CUDACXX) {
    $cmake_args += "-DCMAKE_CUDA_COMPILER:FILEPATH=$Env:CUDACXX"
}

if(-Not $cmake_make_silent) {
    $make_args += "-v"
}

$target_args = @()
if($do_install) {
    $target_args += '--target', 'install'
}

if ([bool]$unparsed_args) {
    $unparsed_args = $unparsed_args | Where-Object {$_ -And $_ -Ne "--%"}
    $unparsed_args = Convert-StringToArgList $unparsed_args `
      | Where-Object {$_ -And $_ -Ne "--%"} `
      | ForEach-Object { $_ -replace "'", '"' } `
      | ForEach-Object { "'$_'" }
    $cmake_args += $unparsed_args
}

# ------------------------------------------------------------------------------

if ([bool]$enable_gpu) {
    # Older CMake using find_package(CUDA) would rely on CUDA_HOME, but newer CMake only look at CUDACXX and CUDA_PATH
    if ([bool]$Env:CUDA_HOME -And -Not [bool]$Env:CUDA_PATH) {
        Write-Output 'CUDA_HOME is defined, but CUDA_PATH is not. Setting CUDA_PATH=CUDA_HOME'
        Push-EnvironmentVariable 'CUDA_PATH' "$Env:CUDA_HOME"
    }
    Write-Debug "CUDA_PATH = $Env:CUDA_PATH"
}

# ------------------------------------------------------------------------------
# Build

if (-Not (Test-Path -Path "$build_dir" -PathType Container) -or $do_clean_build_dir) {
    $do_configure = $true
}
elseif ($do_clean_cache) {
    $do_configure = $true
    Write-Output "Removing CMake cache at: $build_dir/CMakeCache.txt"
    Call-Cmd Remove-Item -Force "'$build_dir/CMakeCache.txt'" -ErrorAction SilentlyContinue
    Write-Output "Removing CMake files at: $build_dir/CMakeFiles"
    Call-Cmd Remove-Item -Force -Recurse "'$build_dir/CMakeFiles'" -ErrorAction SilentlyContinue
    Write-Output "Removing CMake files at: $build_dir/cmake-ldtest*"
    Call-Cmd Remove-Item -Force -Recurse "'$build_dir/cmake-ldtest*'" -ErrorAction SilentlyContinue
}

if ($do_configure) {
    Call-CMake -S "'$source_dir'" -B "'$build_dir'" @cmake_args
}

if ($configure_only) {
    exit 0
}

if ($do_clean) {
    Call-CMake --build "'$build_dir'" --target clean
}

if($do_docs) {
    Call-CMake --build "'$build_dir'" --config "'$build_type'" --target docs @make_args
}

Call-CMake --build "'$build_dir'" --config "'$build_type'" @target_args @make_args

# ==============================================================================

Pop-AllEnvironmentVariables

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

.PARAMETER CCache
If ccache or sccache are found within the PATH, use them with CMake

.PARAMETER CMakeNoRegistry
Disable the use of CMake package registries during configuration

.PARAMETER Clean
Run make clean before building

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

.PARAMETER Configure
Force running the CMake configure step

.PARAMETER Config
Path to INI configuration file with default values for the parameters

.PARAMETER ConfigureOnly
Stop after the CMake configure and generation steps (ie. before building MindQuantum)

.PARAMETER Debug
Build in debug mode

.PARAMETER DebugCMake
Enable debugging mode for CMake configuration step

.PARAMETER DryRun
Dry run; only print commands but do not execute them

.PARAMETER Doc
Setup the Python virtualenv for building the documentation and ask CMake to build the documentation

.PARAMETER Gitee
Use Gitee (where possible) instead of Github/Gitlab

.PARAMETER Gpu
Enable GPU support

.PARAMETER Help
Show help message.

.PARAMETER Install
Build the `install` target

.PARAMETER Jobs
Number of parallel jobs for building

.PARAMETER LocalPkgs
Compile third-party dependencies locally

.PARAMETER Logging
Enable logging in C++ code

.PARAMETER LoggingDebug
Enable DEBUG level logging macros (implies -Logging)

.PARAMETER LoggingTrace
Enable TRACE level logging macros (implies -Logging -LoggingDebug)

.PARAMETER Ninja
Build using Ninja instead of make

.PARAMETER NoConfig
Ignore any configuration file

.PARAMETER NoGitee
Do not favor Gitee over Github/Gitlab

.PARAMETER OnlyPytest
Only install pytest and its dependencies when creating/building the virtualenv

.PARAMETER Prefix
Specify installation prefix

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

PS> .\build_locally.ps1

.EXAMPLE

PS> .\build_locally.ps1 -gpu

.EXAMPLE

PS> .\build_locally.ps1 -Cxx -WithBoost -WithoutEigen3

.EXAMPLE

PS> .\build_locally.ps1 -Gpu -DCMAKE_CUDA_COMPILER=D:\cuda\bin\nvcc
#>
