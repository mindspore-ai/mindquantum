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

if DEFINED build_type set build_type=Relea=
if DEFINED cmake_debug_mode set cmake_debug_mode=
if DEFINED cmake_generator set cmake_generato=
if DEFINED cmake_make_silent set cmake_make_silent=
if DEFINED cuda_arch set cuda_arc=
if DEFINED do_clean_3rdparty set do_clean_3rdparty=
if DEFINED do_clean_build_dir set do_clean_build_dir=
if DEFINED do_clean_cache set do_clean_cache=
if DEFINED do_clean_venv set do_clean_venv=
if DEFINED do_update_venv set do_update_venv=
if DEFINED dry_run set dry_run=
if DEFINED enable_ccache set enable_ccache=
if DEFINED enable_cxx set enable_cxx=
if DEFINED enable_gpu set enable_gpu=
if DEFINED enable_projectq set enable_projectq=
if DEFINED enable_tests set enable_tests=
if DEFINED force_local_pkgs set force_local_pkgs=
if DEFINED local_pkgs set local_pkg=
if DEFINED n_jobs set n_jobs=
if DEFINED prefix_dir set prefix_dir=
if DEFINED only_install_pytest set only_install_pytest=
if DEFINED verbose set verbose=

if DEFINED n_jobs_default set n_jobs_default=

rem ============================================================================

if DEFINED source_dir set source_dir=
if DEFINED build_dir set build_dir=
if DEFINED python_venv_path set python_venv_path=

if DEFINED third_party_libraries (
   set third_party_libraries=
   set third_party_libraries_N=
)

rem ============================================================================
rem Other helper variables

if DEFINED cmake_from_venv set cmake_from_venv=
if DEFINED PYTHON set PYTHON=
