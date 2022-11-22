@rem Copyright 2022 Huawei Technologies Co., Ltd
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

set BASEPATH=%~dp0

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

if !do_clean_venv! == 1 (
  echo Deleting virtualenv folder: !python_venv_path!
  if exist !python_venv_path! call %BASEPATH%\dos\call_cmd.bat rd /Q /S !python_venv_path!
)

rem ----------------------------------------------------------------------------


set venv_args=!python_venv_path!
if "%VENV_USE_SYSTEM_PACKAGES%" == "1" set venv_args=!venv_args! --system-site-packages

set created_venv=0
if NOT EXIST !python_venv_path! (
  set created_venv=1
  echo Creating Python virtualenv: !PYTHON! -m venv !venv_args!
  call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m venv !venv_args!
  goto :activate_venv
)

if !do_update_venv! == 1 (
  set venv_args=!venv_args! --upgrade
  echo Updating Python virtualenv: !PYTHON! -m venv !venv_args!
  call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m venv !venv_args!
)

:activate_venv

echo Activating Python virtual environment: !python_venv_path!
if EXIST !python_venv_path!\Scripts\activate.bat (
  call %BASEPATH%\dos\call_cmd.bat !python_venv_path!\Scripts\activate.bat
) else (
  echo "Unable to activate Python virtual environment!"
  exit /B 1
)

rem ============================================================================

if EXIST !python_venv_path!\bin\!PYTHON! (
   goto :done_adjust_python
)
if EXIST !python_venv_path!\bin\!PYTHON!.exe (
   goto :done_adjust_python
)
if EXIST !python_venv_path!\Scripts\!PYTHON! (
   goto :done_adjust_python
)
if EXIST !python_venv_path!\Scripts\!PYTHON!.exe (
   goto :done_adjust_python
)

:adjust_python

echo !PYTHON! not found in !VIRTUAL_ENV!
echo   -> looking for Pythone executables in !VIRTUAL_ENV!

if EXIST !python_venv_path!\bin\python3 (
   set PYTHON=python3
   goto :done_adjust_python
)
if EXIST !python_venv_path!\bin\python (
   set PYTHON=python
   goto :done_adjust_python
)
if EXIST !python_venv_path!\Scripts\python3.exe (
   set PYTHON=python3.exe
   goto :done_adjust_python
)
if EXIST !python_venv_path!\Scripts\python.exe (
   set PYTHON=python.exe
   goto :done_adjust_python
)

echo "Unable to locate python or python3 in virtual environment!"
exit /B 1

:done_adjust_python

rem ============================================================================
