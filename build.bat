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
@echo off
@title mindquantum_build

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%\build
SET OUTPUT=%BASE_PATH%\output

IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)

rem ============================================================================

cd %BASE_PATH%

rem -----------------------------------------------------------------------------
rem Create a virtual environment for building the wheel

python -m venv venv
call venv\Scripts\activate.bat
python -m pip install -U pip setuptools build wheel pybind11

rem ============================================================================
rem Build the wheels

echo python -m build -w -C--global-option=--set -C--global-option=ENABLE_PROJECTQ %*
python -m build -w -C--global-option=--set -C--global-option=ENABLE_PROJECTQ %*

rem -----------------------------------------------------------------------------
rem Move the wheels to the output directory

IF NOT EXIST "%OUTPUT%" (
    md "output"
)

move /Y %BASE_PATH%\dist\* %OUTPUT%
echo ------Successfully created mindquantum package------
