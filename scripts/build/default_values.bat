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
rem ============================================================================
rem Default values for input arguments

if NOT DEFINED build_type set build_type=Release
if NOT DEFINED cmake_debug_mode set cmake_debug_mode=0
if NOT DEFINED cmake_generator set cmake_generator=
if NOT DEFINED cmake_make_silent set cmake_make_silent=0
if NOT DEFINED cuda_arch set cuda_arch=
if NOT DEFINED do_clean_3rdparty set do_clean_3rdparty=0
if NOT DEFINED do_clean_build_dir set do_clean_build_dir=0
if NOT DEFINED do_clean_cache set do_clean_cache=0
if NOT DEFINED do_clean_venv set do_clean_venv=0
if NOT DEFINED do_update_venv set do_update_venv=0
if NOT DEFINED dry_run set dry_run=0
if NOT DEFINED enable_analyzer set enable_analyzer=0
if NOT DEFINED enable_ccache set enable_ccache=0
if NOT DEFINED enable_cxx set enable_cxx=0
if NOT DEFINED enable_gitee set enable_gitee=0
if NOT DEFINED enable_gpu set enable_gpu=0
if NOT DEFINED enable_projectq set enable_projectq=0
if NOT DEFINED enable_tests set enable_tests=0
if NOT DEFINED enable_logging set enable_logging=0
if NOT DEFINED logging_enable_debug set logging_enable_debug=0
if NOT DEFINED logging_enable_trace set logging_enable_trace=0
if NOT DEFINED force_local_pkgs set force_local_pkgs=0
if NOT DEFINED local_pkgs set local_pkgs=
if NOT DEFINED n_jobs set n_jobs=-1
if NOT DEFINED only_install_pytest set only_install_pytest=0
if NOT DEFINED verbose set verbose=0

if NOT DEFINED n_jobs_default (
   for /f  "tokens=2 delims==" %%d in ('wmic cpu get NumberOfLogicalProcessors /value ^| findstr "="') do @set /A n_jobs_default+=%%d >NUL
)

rem ============================================================================

if NOT DEFINED source_dir set source_dir=!ROOTDIR!
if NOT DEFINED build_dir set build_dir=!source_dir!\build
if NOT DEFINED python_venv_path set python_venv_path=!source_dir!\venv

if NOT DEFINED third_party_libraries (
   set third_party_libraries=boost catch2 eigen3 fmt gmp nlohmann_json projectq pybind11 symengine tweedledum
   set third_party_libraries_N=10
)

rem ============================================================================
rem Other helper variables

if NOT DEFINED cmake_from_venv set cmake_from_venv=0
if NOT DEFINED ninja_from_venv set ninja_from_venv=0
