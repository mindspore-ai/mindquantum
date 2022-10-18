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
set has_ninja=0
set ninja_from_venv=0

if exist !python_venv_path!\Scripts\ninja.exe (
   set NINJA=!python_venv_path!\Scripts\ninja.exe
   set ninja_from_venv=1
   goto: done_ninja
) else (
  if exist !python_venv_path!\bin\ninja.exe (
     set NINJA=!python_venv_path!\bin\ninja.exe
     set ninja_from_venv=1
     goto: done_ninja
  )
)

where ninja >NUL
if %ERRORLEVEL% == 0 (
   set NINJA=ninja
   set has_ninja=1
   goto :done_ninja
)

:install_ninja

set pip_args=
if %_IS_MINDSPORE_CI% == 1 set pip_args=!pip_args! -i https://mirror.baidu.com/pypi/simple

echo Installing Ninja inside the Python virtual environment
call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m pip install !pip_args! ninja

:done_ninja
