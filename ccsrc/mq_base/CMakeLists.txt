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

set(MQ_BASE_HEADERS
    gate/basic_gate.h
    gate/gates.h
    hamiltonian/hamiltonian.h
    matrix/two_dim_matrix.h
    core/popcnt.h
    pr/parameter_resolver.h
    projector/projector.h
    sparse/algo.h
    sparse/csrhdmatrix.h
    sparse/paulimat.h
    sparse/sparse_utils.h
    core/type.h
    core/utils.h)
set(MQ_BASE_SOURCES utils.cc)

if(ENABLE_CUDA)
  add_library(mq_base STATIC ${MQ_BASE_SOURCES})
  target_compile_definitions(mq_base PUBLIC GPUACCELERATED)
  target_link_libraries(mq_base PUBLIC $<IF:$<BOOL:${CUDA_STATIC}>,CUDA::cudart_static,CUDA::cudart>)
else()
  add_library(mq_base STATIC ${MQ_BASE_SOURCES})
endif()
force_at_least_cxx17_workaround(mq_base)
target_sources(mq_base PRIVATE $<BUILD_INTERFACE:${MQ_BASE_HEADERS}>)
target_include_directories(mq_base PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                          $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)
target_link_libraries(mq_base PUBLIC mindquantum_setup ${MQ_OPENMP_TARGET})
append_to_property(mq_install_targets GLOBAL mq_base)

# ==============================================================================

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/core
            ${CMAKE_CURRENT_LIST_DIR}/gate
            ${CMAKE_CURRENT_LIST_DIR}/hamiltonian
            ${CMAKE_CURRENT_LIST_DIR}/matrix
            ${CMAKE_CURRENT_LIST_DIR}/pr
            ${CMAKE_CURRENT_LIST_DIR}/projector
            ${CMAKE_CURRENT_LIST_DIR}/sparse
  DESTINATION ${MQ_INSTALL_INCLUDEDIR})

# ==============================================================================
