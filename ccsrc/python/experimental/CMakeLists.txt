# ==============================================================================
#
# Copyright 2020 <Huawei Technologies Co., Ltd>
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

add_library(mq_cxx_python_interface INTERFACE)
target_include_directories(mq_cxx_python_interface INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                                             $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)
append_to_property(mq_install_targets GLOBAL mq_cxx_python_interface)

# ------------------------------------------------------------------------------

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/python DESTINATION ${MQ_INSTALL_INCLUDEDIR})

# ==============================================================================

pybind11_add_module(
  _mindquantum_cxx
  ${CMAKE_CURRENT_LIST_DIR}/lib/_mindquantum_cxx.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/circuit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/symengine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/ops/ops.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/simulator/simulator.cpp
  MODULE
  OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}/experimental")
target_link_libraries(_mindquantum_cxx PRIVATE mq_cxx_nextgen mq_cxx_python_interface mq_python_core)

# ==============================================================================
