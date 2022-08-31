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

include(CMakePrintHelpers)
cmake_print_properties(TARGETS mq_base PROPERTIES INTERFACE_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)

target_sources(mq_base PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utils.cc)

if(ENABLE_CUDA)
  target_compile_definitions(mq_base PUBLIC GPUACCELERATED)
  target_link_libraries(mq_base PUBLIC $<IF:$<BOOL:${CUDA_STATIC}>,CUDA::cudart_static,CUDA::cudart>)
endif()

target_link_libraries(mq_base PUBLIC ${MQ_OPENMP_TARGET} mindquantum::json)

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
