rem Copyright 2022 Huawei Technologies Co., Ltd
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

set BASEPATH=%~dp0

rem ----------------------------------------------------------------------------

call %BASEPATH%\default_values.bat

rem ============================================================================

if "!ROOTDIR!" == "" (
   echo "(internal error): ROOTDIR variable not defined!"
   exit /B 1
)
if "!PYTHON!" == "" (
   echo "(internal error): PYTHON variable not defined!"
   exit /B 1
)
if "!python_venv_path!" == "" (
   echo "(internal error): python_venv_path variable not defined!"
   exit /B 1
)

rem ============================================================================

rem If from the virtual environment, it's always good
set has_cmake=0
set cmake_from_venv=0

if exist !python_venv_path!\Scripts\cmake.exe (
   set CMAKE=!python_venv_path!\Scripts\cmake.exe
   set cmake_from_venv=1
   goto :done_cmake
) else (
  if exist !python_venv_path!\bin\cmake.exe (
     set CMAKE=!python_venv_path!\bin\cmake.exe
     set cmake_from_venv=1
     goto :done_cmake
  )
)

rem -------------------------------------

type nul > tmp
fc tmp "%ROOTDIR%\CMakeLists.txt" /lb 30 | find "cmake_minimum_required" > tmp
for /F "tokens=2 delims=()" %%i in ('type tmp') do echo %%i > tmp
for /F "tokens=2 delims= " %%i in ('type tmp') do echo %%i > tmp
for /F "tokens=1,2 delims=." %%i in ('type tmp') do (
  set cmake_version_min=%%i.%%j
  set cmake_major_min=%%i
  set cmake_minor_min=%%j
)
del tmp

where cmake >NUL
if %ERRORLEVEL% == 0 (
   set CMAKE=cmake
   goto :has_cmake
)
where cmake3 >NUL
if %ERRORLEVEL% == 0 (
   set CMAKE=cmake3
   goto :has_cmake
)

goto :install_cmake

:has_cmake

for /F "tokens=*" %%i in ('cmake --version') do (
  set cmake_version_str=%%i
  goto :done_get_cmake_version
)

:done_get_cmake_version

for %%i in (!cmake_version_str!) do set cmake_version=%%i

for /F "tokens=1,2 delims=." %%a in ("!cmake_version!") do (
    set cmake_major=%%a
    set cmake_minor=%%b
)

if !cmake_major_min! LEQ !cmake_major! (
   if !cmake_minor_min! LEQ !cmake_minor! (
      goto :done_cmake
   )
)

:install_cmake

set pip_args=--prefer-binary
if %_IS_MINDSPORE_CI% == 1 set pip_args=!pip_args! -i https://mirror.baidu.com/pypi/simple

echo Installing CMake inside the Python virtual environment
call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m pip install !pip_args! "cmake>=!cmake_version_min!"

:done_cmake
