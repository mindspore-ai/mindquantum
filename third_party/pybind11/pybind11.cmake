# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# Single source of truth for the pybind11 version: mindquantum_add_pkg() accepts an installed pybind11 only if it is at
# least VER and otherwise builds VER itself, so VER is both the floor and the fallback. The pyproject.toml build
# requirement mirrors it so that pip fails at resolution time instead of CMake rejecting what pip installed.
set(VER 3.1.0)

if(DEFINED ENV{CODEHUB_CI} AND "$ENV{CODEHUB_CI}" STREQUAL "1")
  set(REQ_URL "$ENV{CODEHUB_PKG_URL}/pybind11-${VER}.tar.gz")
  set(MD5 "235664b4673257b80a9ab79f9b208e94")
elseif(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v${VER}.tar.gz")
  set(MD5 "6e44623bd16e8a2628f11c49943c0842")
else()
  set(REQ_URL "https://github.com/pybind/pybind11/archive/v${VER}.tar.gz")
  set(MD5 "235664b4673257b80a9ab79f9b208e94")
endif()

set(CMAKE_OPTION -DPYBIND11_TEST=OFF)

if(WIN32)
  set(TARGET_ALIAS_EXTRA TARGET_ALIAS mindquantum::windows_extra pybind11::windows_extras)
endif()

# cmake-lint: disable=C0103
set(pybinbd11_LOCAL_EXTRA_DEFINES)
set(pybinbd11_SYSTEM_EXTRA_DEFINES)

if(POSIX_C_SOURCE)
  list(APPEND pybinbd11_LOCAL_EXTRA_DEFINES TARGET pybind11::headers "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  list(APPEND pybinbd11_SYSTEM_EXTRA_DEFINES TARGET pybind11::headers "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
endif()

# cmake-lint: disable=E1122
mindquantum_add_pkg(
  pybind11
  LIBS module headers pybind11 lto pybind11_headers python_headers
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_PKG_NO_COMPONENTS
  CMAKE_OPTION ${CMAKE_OPTION}
  LOCAL_EXTRA_DEFINES ${pybinbd11_LOCAL_EXTRA_DEFINES}
  SYSTEM_EXTRA_DEFINES ${pybinbd11_SYSTEM_EXTRA_DEFINES}
  TARGET_ALIAS mindquantum::pybind11_headers pybind11::headers
  TARGET_ALIAS mindquantum::pybind11_module pybind11::module
  TARGET_ALIAS mindquantum::pybind11_lto pybind11::lto ${TARGET_ALIAS_EXTRA}) # cmake-lint: disable=E1122
