# ==============================================================================
#
# Copyright 2025 <Huawei Technologies Co., Ltd>
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
#
# lint_cmake: -whitespace/indent
#
# cmake-lint: disable=C0103

if(NOT ENABLE_CUDA)
  return()
endif()

# Determine CPU architecture for cuQuantum archive naming
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(_CQ_ARCH linux-x86_64)
else()
  message(FATAL_ERROR "Unsupported architecture for cuQuantum: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Extract CUDA major version (e.g., "12" from "12.0")
string(REGEX MATCH "^[0-9]+" _CQ_CUDA_MAJOR "${MQ_CUDA_VERSION}")

set(VER 25.06.0.10)
set(REQ_URL "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/${_CQ_ARCH}/cuquantum-${_CQ_ARCH}-${VER}_cuda${_CQ_CUDA_MAJOR}-archive.tar.xz")
if(_CQ_CUDA_MAJOR STREQUAL "11")
  set(MD5 "6d329713ddb3ced0d08adbc4010a637a")
elseif(_CQ_CUDA_MAJOR STREQUAL "12")
  set(MD5 "9eaf4561800c44c4a29890ea59e2d5a0")
else()
  message(FATAL_ERROR
    "Unsupported cuQuantum CUDA major version: ${_CQ_CUDA_MAJOR}"
  )
endif()

mindquantum_add_pkg(
  cuquantum
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  FORCE_LOCAL_PKG
  ONLY_COPY_DIRS include
  TARGET_ALIAS mindquantum::cuquantum cuquantum::cuquantum)

# Expose cuQuantum headers (from <prefix>/include) on the imported interface.
if(TARGET mindquantum::cuquantum)
  target_include_directories(cuquantum::cuquantum INTERFACE
    $<BUILD_INTERFACE:${cuquantum_BASE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
endif()
