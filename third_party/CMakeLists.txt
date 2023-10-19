# ==============================================================================
#
# Copyright 2021 <Huawei Technologies Co., Ltd>
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

# lint_cmake: -whitespace/indent

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_CURRENT_LIST_DIR}/cmake/modules)

# ------------------------------------------------------------------------------

include(utils)
include(external_targets)

# =============================================================================

set(POSIX_C_SOURCE 0)
if("${OS_NAME}" STREQUAL "MSYS-MSYS" OR CYGWIN)
  # 200809 because of strdup() or strnlen() (otherwise 200112 for posix_memalign()
  set(POSIX_C_SOURCE 200809)
endif()

# ------------------------------------------------------------------------------

include(${CMAKE_CURRENT_LIST_DIR}/fmt/fmt.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/pybind11/pybind11.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/nlohmann_json/nlohmann_json.cmake)

if(ENABLE_LOGGING)
  include(${CMAKE_CURRENT_LIST_DIR}/spdlog/spdlog.cmake)
endif()

if(BUILD_TESTING)
  include(${CMAKE_CURRENT_LIST_DIR}/catch2/catch2.cmake)
endif()

# ==============================================================================

get_property(_pkgs GLOBAL PROPERTY mq_external_packages)

set(_external_pkgs_str)
foreach(_pkg ${_pkgs})
  load_from_cache(_${_pkg}_find_pkg_str _tmp)
  set(_external_pkgs_str "${_external_pkgs_str}\n\n${_tmp}")
  message(STATUS ${_tmp})
endforeach()

# get_property(_tmp GLOBAL PROPERTY mq_external_find_packages) string(REPLACE ";" "\n" _tmp "${_tmp}")

# --------------------------------------

set(MQ_3RDPARTY_PREFIX_PATH "\${MQ_PACKAGE_PREFIX_DIR}/${MQ_INSTALL_3RDPARTYDIR}")
string(CONFIGURE "${_external_pkgs_str}" MQ_FIND_EXTERNAL_PACKAGES)
configure_file("${CMAKE_CURRENT_LIST_DIR}/cmake/mindquantumTargetsExternal.cmake.in"
               "${PROJECT_BINARY_DIR}/config_for_install/mindquantumTargetsExternal.cmake" @ONLY)

install(FILES ${PROJECT_BINARY_DIR}/config_for_install/mindquantumTargetsExternal.cmake
        DESTINATION ${MQ_INSTALL_CMAKEDIR})

# --------------------------------------

set(MQ_3RDPARTY_PREFIX_PATH "${_mq_local_prefix}")
string(CONFIGURE "${_external_pkgs_str}" MQ_FIND_EXTERNAL_PACKAGES)
configure_file("${CMAKE_CURRENT_LIST_DIR}/cmake/mindquantumTargetsExternal.cmake.in"
               "${PROJECT_BINARY_DIR}/mindquantumTargetsExternal.cmake" @ONLY)

# ==============================================================================

list(POP_FRONT CMAKE_MODULE_PATH)
list(POP_FRONT CMAKE_MODULE_PATH)
