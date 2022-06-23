@echo off
@title mindquantum_build

@rem Copyright 2020 Huawei Technologies Co., Ltd
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem ============================================================================

setlocal ENABLEDELAYEDEXPANSION ENABLEEXTENSIONS

set BASEPATH=%~dp0
set ROOTDIR=%BASEPATH%
set SCRIPTDIR=%BASEPATH%\scripts\build
set PROGRAM=%~nx0

rem ============================================================================
rem Default values for this particular script

set enable_gitee=1
set has_build_dir=0
set delocate_wheel=1
set build_isolation=1
set output_path=%ROOTDIR%\output
set platform_name=
set python_extra_pkgs=setuptools-scm[toml] wheel-filename>1.2

call %SCRIPTDIR%\default_values.bat

rem ============================================================================

:initial
  set result=false
  if "%1" == "" goto :done_parsing

  if /I "%1" == "/h" set result=true
  if /I "%1" == "/Help" set result=true
  if "%result%" == "true" (
    call :help_message
    goto :END
  )

  if /I "%1" == "/N" (
    set dry_run=1
    shift & goto :initial
  )

  if /I "%1" == "/B" set result=true
  if /I "%1" == "/Build" set result=true
  if "%result%" == "true" (
    set value=%2
    if not defined value goto :arg_build
    if "!value:~0,1!" == "/" (
      :arg_build
      echo %PROGRAM%: option requires an argument -- '/B,/Build'
      goto :END
    )
    set has_build_dir=1
    set build_dir=!value!
    shift & shift & goto :initial
  )

  if /I "%1" == "/Clean" (
    set do_clean=1
    shift & goto :initial
  )
  if /I "%1" == "/Clean3rdParty" (
    set do_clean_3rdparty=1
    shift & goto :initial
  )
  if /I "%1" == "/CleanAll" (
    set do_clean_venv=1
    set do_clean_build_dir=1
    shift & goto :initial
  )
  if /I "%1" == "/CleanCache" (
    set do_clean_cache=1
    shift & goto :initial
  )
  if /I "%1" == "/CleanVenv" (
    set do_clean_venv=1
    shift & goto :initial
  )

  if /I "%1" == "/C" set result=true
  if /I "%1" == "/Configure" set result=true
  if "%result%" == "true" (
    set do_configure=1
    shift & goto :initial
  )
  if /I "%1" == "/ConfigureOnly" (
    set configure_only=1
    shift & goto :initial
  )

  if /I "%1" == "/CudaArch" (
    set value=%2
    if not defined value goto :arg_cuda_arch
    if "!value:~0,1!" == "/" (
      :arg_cuda_arch
      echo %PROGRAM%: option requires an argument -- '/CudaArch'
      goto :END
    )
    call :ToCMakeList value
    set cuda_arch=!value!
    shift & shift & goto :initial
  )

  if /I "%1" == "/Cxx" (
    set enable_cxx=1
    shift & goto :initial
  )

  if /I "%1" == "/Debug" (
    set build_type=Debug
    shift & goto :initial
  )

  if /I "%1" == "/DebugCMake" (
    set cmake_debug_mode=1
    shift & goto :initial
  )

  if /I "%1" == "/Delocate" (
    set delocate_wheel=1
    shift & goto :initial
  )
  if /I "%1" == "/NoDelocate" (
    set delocate_wheel=0
    shift & goto :initial
  )

  if /I "%1" == "/Gpu" (
    set enable_gpu=1
    shift & goto :initial
  )

  if /I "%1" == "/Install" (
    set do_install=1
    shift & goto :initial
  )

  if /I "%1" == "/J" set result=true
  if /I "%1" == "/Jobs" set result=true
  if "%result%" == "true" (
    set value=%2
    if not defined value goto :arg_build
    if "!value:~0,1!" == "/" (
      :arg_build
      echo %PROGRAM%: option requires an argument -- '/B,/Build'
      goto :END
    )
    set n_jobs=!value!
    shift & shift & goto :initial
  )

  if /I "%1" == "/LocalPkgs" (
    set force_local_pkgs=1
    shift & goto :initial
  )

  if /I "%1" == "/Ninja" (
    set ninja=1
    shift & goto :initial
  )

  if /I "%1" == "/OnlyPytest" (
    set install_only_pytest=1
    shift & goto :initial
  )

  if /I "%1" == "/NoIsolation" (
    set build_isolation=0
    shift & goto :initial
  )

  if /I "%1" == "/Prefix" (
    set value=%2
    if not defined value goto :arg_prefix
    if "!value:~0,1!" == "/" (
      :arg_prefix
      echo %PROGRAM%: option requires an argument -- '/Prefix'
      goto :END
    )
    set prefix_dir=!value!
    shift & shift & goto :initial
  )

  if /I "%1" == "/Quiet" (
    set cmake_make_silent=1
    shift & goto :initial
  )

  if /I "%1" == "/ShowLibraries" (
    call :print_show_libraries
    goto :END
  )

  if /I "%1" == "/Test" (
    set enable_tests=1
    shift & goto :initial
  )

  if /I "%1" == "/UpdateVenv" (
    set do_update_venv=1
    shift & goto :initial
  )

  if /I "%1" == "/Venv" (
    set value=%2
    if not defined value goto :arg_venv
    if "!value:~0,1!" == "/" (
      :arg_venv
      echo %PROGRAM%: option requires an argument -- '/Venv'
      goto :END
    )
    set python_venv_path=!value!
    shift & shift & goto :initial
  )

  set value=%1
  set with_header=!value:~0,5!
  if /I "!with_header!" == "/with" (
    set library=!value:~5!
    call :LoCase library
    if not defined local_pkgs (
      set local_pkgs=!library!
    ) else (
      set local_pkgs=!local_pkgs!,!library!
    )
    shift & goto :initial
  )

  set unparsed_args=!unparsed_args! %1
  shift & goto :initial

:done_parsing

rem ============================================================================
rem Locate python or python3

call %SCRIPTDIR%\locate_python3.bat
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%


rem ============================================================================

cd %ROOTDIR%

rem ----------------------------------------------------------------------------

call %SCRIPTDIR%\python_virtualenv_activate.bat
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

rem ------------------------------------------------------------------------------
rem Locate cmake or cmake3

call %SCRIPTDIR%\locate_cmake.bat
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

rem ----------------------------------------------------------------------------

call %SCRIPTDIR%\python_virtualenv_update.bat
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

rem ============================================================================
rem Setup arguments for build

set args=-w

if !build_isolation! == 0 set args=!args! --no-isolation

set RETVAL=
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_CMAKE_DEBUG !cmake_debug_mode!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_CXX_EXPERIMENTAL !enable_cxx!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_CUDA !enable_gpu!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_GITEE !enable_gitee!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_PROJECTQ !enable_projectq!
call %SCRIPTDIR%\dos\build_cmake_option.bat BUILD_TESTING !enable_tests!
call %SCRIPTDIR%\dos\build_cmake_option.bat CLEAN_3RDPARTY_INSTALL_DIR !do_clean_3rdparty!

set args=!args! %RETVAL%

if !cmake_make_silent! == 0 (
   set args=!args! -C--global-option=--set -C--global-option=USE_VERBOSE_MAKEFILE
) else (
   set args=!args! -C--global-option=--unset -C--global-option=USE_VERBOSE_MAKEFILE
)

if !ninja! == 1 (
   set args=!args! -C--global-option=-GNinja
) else (
   if !n_jobs! == -1 set n_jobs=!n_jobs_default!
)

if NOT !n_jobs! == -1 set args=!args! -C--global-option=build -C--global-option=--parallel  -C--global-option=!n_jobs!

if "!build_type!" == "Debug" set args=!args! -C--global-option=build -C--global-option=--debug

if !force_local_pkgs! == 1 (
  set args=!args! -C--global-option=--var -C--global-option=MQ_FORCE_LOCAL_PKGS -C--global-option=all
) else (
  if NOT "!local_pkgs!" == "" (
    set args=!args! -C--global-option=--var -C--global-option=MQ_FORCE_LOCAL_PKGS -C--global-option=!local_pkgs!
  )
)

if !has_build_dir! == 1 set args=!args! -C--global-option=build_ext -C--global-option=--build-dir -C--global-option=!build_dir!

if NOT "!CC!" == "" set args=!args! -C--global-option=--var -C--global-option=CMAKE_C_COMPILER -C--global-option=!CC!
if NOT "!CXX!" == "" set args=!args! -C--global-option=--var -C--global-option=CMAKE_CXX_COMPILER -C--global-option=!CXX!
if NOT "!CUDACXX!" == "" set args=!args! -C--global-option=--var -C--global-option=CMAKE_CUDA_COMPILER -C--global-option=!CUDACXX!

rem ============================================================================
rem Build the wheels

if !has_build_dir! == 1 (
  if !do_clean_build_dir! == 1 (
    echo Deleting build folder: !build_dir!
    if exist !build_dir! call :call_cmd rd /Q /S !build_dir!
  ) else (
    if !do_clean_cache! == 1 (
      echo Removing CMake cache at: !build_dir!\CMakeCache.txt
      if exist !build_dir!\CMakeCache.txt call :call_cmd del /Q "!build_dir!\CMakeCache.txt"
      echo Removing CMake files at: !build_dir!/CMakeFiles
      if exist !build_dir!/CMakeFiles call :call_cmd rd /Q /S "!build_dir!\CMakeFiles"
    )
  )
)

set MQ_DELOCATE_WHEEL=!delocate_wheel!
set MQ_DELOCATE_WHEEL_PLAT=
if NOT "!platform_name!" == "" set MQ_DELOCATE_WHEEL_PLAT=!platform_name!

call %SCRIPTDIR%\dos\call_cmd.bat !PYTHON! -m build !args! !unparsed_args!

if DEFINED args set args=
if DEFINED unparsed_args set unparsed_args=

if DEFINED MQ_DELOCATE_WHEEL set MQ_DELOCATE_WHEEL=
if DEFINED MQ_DELOCATE_WHEEL_PLAT set MQ_DELOCATE_WHEEL_PLAT=

rem -----------------------------------------------------------------------------
rem Move the wheels to the output directory

IF NOT EXIST "!output_path!" (
    md "!output_path!"
)

call %SCRIPTDIR%\dos\call_cmd.bat move /Y %ROOTDIR%\dist\* %output_path%

echo ------Successfully created mindquantum package------

goto :END

rem ============================================================================

:help_message
  echo Build binary Python wheel for MindQunantum
  echo:
  echo This is mainly relevant for developers that want to deploy MindQuantum
  echo on machines other than their own.
  echo:
  echo This script will create a Python virtualenv in the MindQuantum root
  echo directory and then build a binary Python wheel of MindQuantum.
  echo:
  echo Usage:
  echo   %PROGRAM% [options]
  echo:
  echo Options:
  echo   /H,/help            Show this help message and exit
  echo   /N                  Dry run; only print commands but do not execute them
  echo
  echo   /B,/Build [dir]     Specify build directory
  echo                       Defaults to: %build_dir%
  echo   /Clean3rdParty      Clean 3rd party installation directory
  echo   /CleanAll           Clean everything before building.
  echo                       Equivalent to /CleanVenv /CleanBuilddir
  echo   /CleanBuildDir      Delete build directory before building
  echo   /CleanCache         Re-run CMake with a clean CMake cache
  echo   /CleanVenv          Delete Python virtualenv before building
  echo   /Cxx                (experimental) Enable MindQuantum C++ support
  echo   /Debug              Build in debug mode
  echo   /Delocate           Delocate the binary wheels after build is finished
  echo                       (enabled by default; pass /NoDelocate to disable)
  echo   /Gpu                Enable GPU support
  echo   /j,/Jobs [N]        Number of parallel jobs for building
  echo                       Defaults to: !n_jobs_default!
  echo   /LocalPkgs          Compile third-party dependencies locally
  echo   /Ninja              Use the Ninja CMake generator
  echo   /NoDelocate         Disable delocating the binary wheels after build is finished
  echo   /NoIsolation        Pass --no-isolation to python3 -m build
  echo   /O, /Output [dir]   Output directory for built wheels
  echo   /PlatName           Platform name to use for wheel delocation
  echo                       (only effective if --delocate is used)
  echo   /Quiet              Disable verbose build rules
  echo   /ShowLibraries      Show all known third-party libraries
  echo   /Venv *path*        Path to Python virtual environment
  echo                       Defaults to: %python_venv_path%
  echo   /With*library*      Build the third-party *library* from source (*library* is case-insensitive)
  echo                       (ignored if /LocalPkgs is passed, except for projectq)
  rem echo   /Without*library*   Do not build the third-party library from source (*library* is case-insensitive)
  rem echo                       (ignored if /LocalPkgs is passed, except for projectq)
  echo:
  echo Test related options:
  echo   /Test               Build C++ tests and install dependencies for Python testing as well
  echo   /OnlyPytest         Only install pytest and its dependencies when creating/building the virtualenv
  echo:
  echo Python related options:
  echo   /UpdateVenv         Update the python virtual environment
  echo:
  echo NB: any unknown arguments will be passed on to 'python3 -m build'
  echo:
  echo Example calls:
  echo %PROGRAM%
  echo %PROGRAM% /gpu
  echo %PROGRAM% /cxx /WithBoost
  echo %PROGRAM% "-DCMAKE_CUDA_COMPILER^=/opt/cuda/bin/nvcc"
  EXIT /B 0

rem ============================================================================

:END

call %SCRIPTDIR%\unset_values.bat
