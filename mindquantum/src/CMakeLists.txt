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

# for pybind
pybind11_add_module(mqbackend ${CMAKE_CURRENT_SOURCE_DIR}/binding.cc)
target_compile_definitions(mqbackend PUBLIC $<$<BOOL:${ENABLE_PROJECTQ}>:ENABLE_PROJECTQ>
                                            $<$<BOOL:${ENABLE_QUEST}>:ENABLE_QUEST>)
target_include_directories(mqbackend PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(mqbackend PUBLIC ${PARALLEL_LIBS} "$<$<BOOL:${ENABLE_PROJECTQ}>:mq_projectq>"
                                       "$<$<BOOL:${ENABLE_QUEST}>:QuEST;mq_quest>" mq_base)

set_output_directory_auto(mqbackend mindquantum)
python_install_set_rpath(mqbackend "")

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
add_subdirectory(backends)

if(ENABLE_CUDA)
  add_library(mq_base STATIC ${MQ_BASE_SOURCES} ${MQ_BASE_HEADERS})

  target_compile_definitions(mq_base PUBLIC GPUACCELERATED)
  target_link_libraries(mq_base PUBLIC $<IF:$<BOOL:${CUDA_STATIC}>,CUDA::cudart_static,CUDA::cudart>)
else()
  add_library(mq_base STATIC ${MQ_BASE_SOURCES} ${MQ_BASE_HEADERS})
endif()
target_include_directories(mq_base PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(WIN32)
  get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
  file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
    ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_*-1.dll ${CXX_DIR}/libgomp-1.dll)
  # install(FILES ${LIB_LIST} DESTINATION ${CMAKE_BINARY_DIR})
  foreach(WIN_DEP_LIB ${LIB_LIST})
    MESSAGE("COPYING ${WIN_DEP_LIB}")
    file(COPY ${WIN_DEP_LIB} DESTINATION ${MQBACKEND_OUTPUT_DIR} FOLLOW_SYMLINK_CHAIN)
  endforeach()
endif()