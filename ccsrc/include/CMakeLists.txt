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

# lint_cmake: -whitespace/indent

set(MQ_BASE_HEADERS
    config/popcnt.hpp
    config/config.hpp
    config/type_traits.hpp
    core/sparse/algo.hpp
    core/sparse/csrhdmatrix.hpp
    core/sparse/paulimat.hpp
    core/sparse/sparse_utils.hpp
    core/mq_base_types.hpp
    core/utils.hpp
    ops/basic_gate.hpp
    ops/gates.hpp
    ops/hamiltonian.hpp
    ops/projector.hpp)
target_sources(mq_base PRIVATE $<BUILD_INTERFACE:${MQ_BASE_HEADERS}>)
target_include_directories(mq_base PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                          $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)

# ==============================================================================

add_library(include_lib INTERFACE)
target_include_directories(include_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                                 $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)
append_to_property(mq_install_targets GLOBAL include_lib)

# ------------------------------------------------------------------------------

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config ${CMAKE_CURRENT_LIST_DIR}/core ${CMAKE_CURRENT_LIST_DIR}/ops
            ${CMAKE_CURRENT_LIST_DIR}/simulator ${CMAKE_CURRENT_LIST_DIR}/device ${CMAKE_CURRENT_LIST_DIR}/math
  DESTINATION ${MQ_INSTALL_INCLUDEDIR}/
  PATTERN "CPPLINT.cfg" EXCLUDE)

# =============================================================================

if(ENABLE_CXX_EXPERIMENTAL)
  add_subdirectory(experimental)
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/experimental DESTINATION ${MQ_INSTALL_INCLUDEDIR}/)
endif()

# =============================================================================
