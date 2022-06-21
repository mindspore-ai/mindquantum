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

# TODO: Create OBJECT libraries to avoid duplicate compilations For that need to get pybind11 dependencies manually...

# ==============================================================================

add_library(mindquantum_python_interface INTERFACE)
target_include_directories(mindquantum_python_interface INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                                                  $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)
append_to_property(mq_install_targets GLOBAL mindquantum_python_interface)

# ------------------------------------------------------------------------------

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/python DESTINATION ${MQ_INSTALL_INCLUDEDIR})

# ==============================================================================

pybind11_add_module(
  _mindquantum_cxx
  ${CMAKE_CURRENT_LIST_DIR}/lib/_mindquantum_cxx.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/circuit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/symengine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/ops/ops.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/optimizer/optimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/simulator/simulator.cpp
  MODULE
  OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}/experimental")
target_link_libraries(_mindquantum_cxx PRIVATE mindquantum_cxx mindquantum_python_interface)

# ==============================================================================

pybind11_add_module(
  _mindquantum_cxx_core
  ${CMAKE_CURRENT_LIST_DIR}/lib/_mindquantum_cxx_core.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/base.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/engine_list.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/optimisation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/printer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/resource_counter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/core/core.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/ops/command.cpp
  MODULE
  OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}/experimental")

target_link_libraries(_mindquantum_cxx_core PRIVATE mindquantum_cxx mindquantum_python_interface)
set_output_directory_auto(_mindquantum_cxx_core mindquantum/experimental)

# ==============================================================================

pybind11_add_module(
  _mindquantum_cxx_cengines
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/base.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/cengines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/engine_list.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/optimisation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/printer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/cengines/resource_counter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/ops/command.cpp
  MODULE
  OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}/experimental")

target_link_libraries(_mindquantum_cxx_cengines PRIVATE mindquantum_cxx mindquantum_python_interface)
set_output_directory_auto(_mindquantum_cxx_cengines mindquantum/experimental)

# ==============================================================================
