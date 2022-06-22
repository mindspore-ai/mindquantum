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

where python3
if %ERRORLEVEL% == 0 (
  set PYTHON=python3
  goto :done_python
)
where python
if %ERRORLEVEL% == 0 (
  set PYTHON=python
  goto :done_python
)

echo "Unable to locate python or python3!"
exit /B 1

:done_python

for /F %%i in ('where !PYTHON!') do (
   set PYTHON_ABS=%%i
   goto :done_where
)
:done_where

call %BASEPATH%\dos\debug_print.bat "Using Python from !PYTHON_ABS!"
set PYTHON_ABS=
