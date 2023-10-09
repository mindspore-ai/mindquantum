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

set _IS_MINDSPORE_CI=0

if NOT "%DEVCLOUD_CI%" == "" goto :CI_DEVCLOUD
if NOT "%CI%" == "" goto :CI_JENKINS

goto :CI_FALSE

:CI_DEVCLOUD
if /I "%DEVCLOUD_CI%" == "true" goto :CI_TRUE
if %DEVCLOUD_CI% == 1 goto :CI_TRUE
goto :CI_FALSE

:CI_JENKINS
if /I "%CI%" == "false" goto :CI_FALSE
if %CI% == 0 goto :CI_FALSE

if NOT %JENKINS_URL% == "" (
   echo %JENKINS_URL% | findstr /r "^https*://build.mindspore.cn">nul 2>&1
   if %errorlevel% == 0 goto :CI_TRUE
)

:CI_FALSE
set _IS_MINDSPORE_CI=0
goto :CI_DONE

:CI_TRUE
echo Detected MindSpore/MindQuantum CI
set _IS_MINDSPORE_CI=1

:CI_DONE

rem ============================================================================
rem Default values for this particular script

set enable_gitee=0
set has_build_dir=0
set delocate_wheel=1
set build_isolation=1
set output_path=%ROOTDIR%\output
set platform_name=
set python_extra_pkgs=wheel-filename>1.2 build==0.9.0

if !_IS_MINDSPORE_CI! == 1 (
   set cmake_debug_mode=1
   set enable_gitee=1
   set enable_gpu=1
)

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

  if /I "%1" == "/analyzer" (
    set enable_analyzer=1
    shift & goto :initial
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

  if /I "%1" == "/BuildIsolation" (
    set build_isolation=1
    shift & goto :initial
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

  if /I "%1" == "/NoGitee" (
    set enable_gitee=0
    shift & goto :initial
  )

  if /I "%1" == "/NoIsolation" (
    set build_isolation=0
    shift & goto :initial
  )

  if /I "%1" == "/Gitee" (
    set enable_gitee=1
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

  if /I "%1" == "/Logging" (
    set enable_logging=1
    shift & goto :initial
  )

  if /I "%1" == "/LoggingDebug" (
    set enable_logging=1
    set logging_enable_debug=1
    shift & goto :initial
  )

  if /I "%1" == "/LoggingTrace" (
    set enable_logging=1
    set logging_enable_trace=1
    shift & goto :initial
  )

  if /I "%1" == "/NoBuildIsolation" (
    set build_isolation=0
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

rem ------------------------------------------------------------------------------
rem Locate ninja if needed

if !ninja! == 1 (
  call %SCRIPTDIR%\locate_ninja.bat
  if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%
)

rem ----------------------------------------------------------------------------

call %SCRIPTDIR%\python_virtualenv_update.bat
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

rem ============================================================================
rem Setup arguments for build

set args=-w

if !build_isolation! == 0 set args=!args! --no-isolation

set RETVAL=
call %SCRIPTDIR%\dos\build_cmake_option.bat BUILD_TESTING !enable_tests!
call %SCRIPTDIR%\dos\build_cmake_option.bat CLEAN_3RDPARTY_INSTALL_DIR !do_clean_3rdparty!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_ANALYZER !enable_analyzer!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_CMAKE_DEBUG !cmake_debug_mode!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_CUDA !enable_gpu!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_GITEE !enable_gitee!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_LOGGING !enable_logging!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_LOGGING_DEBUG_LEVEL !logging_enable_debug!
call %SCRIPTDIR%\dos\build_cmake_option.bat ENABLE_LOGGING_TRACE_LEVEL !logging_enable_trace!

set args=!args! %RETVAL%

if !_IS_MINDSPORE_CI! == 1 (
   set args=!args! -C--global-option=--set -C--global-option=MINDSPORE_CI
)

if !cmake_make_silent! == 0 (
   set args=!args! -C--global-option=--set -C--global-option=USE_VERBOSE_MAKEFILE
) else (
   set args=!args! -C--global-option=--unset -C--global-option=USE_VERBOSE_MAKEFILE
)

if !ninja! == 1 (
   set args=!args! -C--global-option=-G"Ninja Multi-Config"
) else (
   if !n_jobs! == -1 set n_jobs=!n_jobs_default!
)

if NOT !n_jobs! == -1 (
  set args=!args! -C--global-option=--var -C--global-option=JOBS -C--global-option=!n_jobs!
  set args=!args! -C--global-option=build_ext -C--global-option=--jobs=!n_jobs!
)

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

if !enable_gpu! == 1 (
  if "!CUDA_PATH" == "" (
    if NOT "!CUDA_HOME!" == "" (
       rem Older CMake using find_package(CUDA) would rely on CUDA_HOME, but newer CMake only look at CUDACXX and
       rem CUDA_PATH
       echo CUDA_HOME is defined, but CUDA_PATH is not. Setting CUDA_PATH=CUDA_HOME
       set CUDA_PATH=!CUDA_HOME!
    )
  )
  echo CUDA_HOME = !CUDA_HOME!
  echo CUDA_PATH = !CUDA_PATH!
)

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
      echo Removing CMake files at: !build_dir!/cmake-ldtest
      if exist !build_dir!/CMakeFiles call :call_cmd rd /Q /S "!build_dir!\cmake-ldtest"
    )
  )
)

if !delocate_wheel! == 1 (
  set MQ_DELOCATE_WHEEL=1
  set MQ_DELOCATE_WHEEL_PLAT=
  if NOT "!platform_name!" == "" set MQ_DELOCATE_WHEEL_PLAT=!platform_name!

  if !has_build_dir! == 1 (
    set build_dir_for_env=!build_dir!
  ) else (
    if "!fast_build_dir!" == "" (
      for /F "delims=" %%i IN ('!PYTHON! -m mindquantum_config --tempdir') DO set build_dir_for_env=%%i
    ) else (
      set build_dir_for_env=!fast_build_dir!
    )
  )

  if !_IS_MINDSPORE_CI! == 1 (
    set MQ_LIB_PATHS=!ROOTDIR!\ld_library_paths.txt
  ) else (
    set MQ_LIB_PATHS=!build_dir_for_env!\ld_library_paths.txt
  )
  set MQ_BUILD_DIR=!build_dir_for_env!

  echo MQ_LIB_PATHS = !MQ_LIB_PATHS!
  echo MQ_BUILD_DIR = !MQ_BUILD_DIR!
)

call %SCRIPTDIR%\dos\call_cmd.bat !PYTHON! -m build !args! !unparsed_args!

if DEFINED args set args=
if DEFINED unparsed_args set unparsed_args=

if DEFINED MQ_DELOCATE_WHEEL set MQ_DELOCATE_WHEEL=
if DEFINED MQ_DELOCATE_WHEEL_PLAT set MQ_DELOCATE_WHEEL_PLAT=
if DEFINED MQ_LIB_PATHS set MQ_LIB_PATHS=
if DEFINED MQ_BUILD_DIR set MQ_BUILD_DIR=

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
  echo   /Analyzer           Use the compiler static analysis tool during compilation (GCC or MSVC)
  echo   /B,/Build [dir]     Specify build directory
  echo                       Defaults to: %build_dir%
  echo   /Clean3rdParty      Clean 3rd party installation directory
  echo   /CleanAll           Clean everything before building.
  echo                       Equivalent to /CleanVenv /CleanBuilddir
  echo   /CleanBuildDir      Delete build directory before building
  echo   /CleanCache         Re-run CMake with a clean CMake cache
  echo   /CleanVenv          Delete Python virtualenv before building
  echo   /Debug              Build in debug mode
  echo   /Delocate           Delocate the binary wheels after build is finished
  echo                       (enabled by default; pass /NoDelocate to disable)
  echo   /Gitee              Use Gitee (where possible) instead of Github/Gitlab
  echo   /Gpu                Enable GPU support
  echo   /j,/Jobs [N]        Number of parallel jobs for building
  echo                       Defaults to: !n_jobs_default!
  echo   /LocalPkgs          Compile third-party dependencies locally
  echo   /Logging            Enable logging in C++ code
  echo   /LoggingDebug       Enable DEBUG level logging macros (implies /Logging)
  echo   /LoggingTrace       Enable TRACE level logging macros (implies /Logging /LoggingDebug)'
  echo   /Ninja              Use the Ninja CMake generator
  echo   /NoDelocate         Disable delocating the binary wheels after build is finished
  echo   /NoGitee            Do not favor Gitee over Github/Gitlab
  echo   /NoIsolation        Pass --no-isolation to python3 -m build
  echo   /O, /Output [dir]   Output directory for built wheels
  echo   /PlatName           Platform name to use for wheel delocation
  echo                       (only effective if --delocate is used)

  echo   /Quiet              Disable verbose build rules
  echo   /ShowLibraries      Show all known third-party libraries
  echo   /Venv *path*        Path to Python virtual environment
  echo                       Defaults to: %python_venv_path%
  echo   /With*library*      Build the third-party *library* from source (*library* is case-insensitive)
  echo                       (ignored if /LocalPkgs is passed)
  rem echo   /Without*library*   Do not build the third-party library from source (*library* is case-insensitive)
  rem echo                       (ignored if /LocalPkgs is passed)
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
