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

call %BASEPATH%\default_values.bat

rem ============================================================================

if "!PYTHON!" == "" (
   echo "(internal error): PYTHON variable not defined!"
   exit /B 1
)

if "!ROOTDIR!" == "" (
   echo "(internal error): ROOTDIR variable not defined!"
   exit /B 1
)

rem ============================================================================

if !created_venv! == 1 goto :update_venv
if !do_update_venv! == 1 goto :update_venv

goto :EOF

:update_venv

set critical_pkgs=pip "setuptools>=61,<=65.5.1;python_version<'3.12'" "setuptools>=61;python_version>='3.12'" "wheel" "pybind11>=2.9.2" "wheel-filename>1.2" build
echo Updating critical Python packages: !PYTHON! -m pip install -U !critical_pkgs!
call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m pip install -U !critical_pkgs!

set pkgs=pybind11

if !cmake_from_venv! == 1 set pkgs=!pkgs! cmake

if !ninja_from_venv! == 1 set pkgs=!pkgs! ninja

if !enable_tests! == 1 (
  if NOT "%VENV_PYTHON_TEST_PKGS%" == "" (
    set pkgs=!pkgs! %VENV_PYTHON_TEST_PKGS%
    goto :done_test_pkgs
  )

  if !only_install_pytest! == 1 (
    set pkgs=!pkgs! pytest pytest-cov pytest-mock mock
    goto :done_test_pkgs
  )

  call :mktemp tmp_file "%temp%"

  pushd "!ROOTDIR!"
  "!PYTHON!" setup.py gen_reqfile --include-extras test --output "!tmp_file!"
  popd

  for /F %%i in ('findstr /v "^[ ]*$" "!tmp_file!"') do set pkgs=!pkgs! %%i
  del /F !tmp_file!
)

:done_test_pkgs

if !do_docs! == 1 set pkgs=!pkgs! breathe sphinx sphinx_rtd_theme importlib-metadata myst-parser

if NOT "!python_extra_pkgs!" == "" set pkgs=!pkgs! !python_extra_pkgs!

set pip_args=--prefer-binary
if !do_update_venv! == 1 set pip_args=!pip_args! -U

rem  TODO(dnguyen): add wheel delocation package for Windows once we figure this out

echo Updating Python packages: !PYTHON! -m pip install !pip_args! !pkgs!
call %BASEPATH%\dos\call_cmd.bat !PYTHON! -m pip install !pip_args! !pkgs!

set pip_args=
set pkgs=

goto :EOF

rem ============================================================================

:mktemp
  @rem in the next line (optional): create the "%~2" folder if does not exist
  md "%~2" 2>NUL
  set "_uniqueFileName=%~2\bat~%RANDOM%.tmp"
  if exist "%_uniqueFileName%" goto :uniqGet
  set "%~1=%_uniqueFileName%"
  exit /B 0
