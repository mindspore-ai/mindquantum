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

pybind11_add_module(_mq_vector MODULE OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}"
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector.cpp)

target_include_directories(_mq_vector PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

force_at_least_cxx17_workaround(_mq_vector)
target_link_libraries(_mq_vector PUBLIC mqsim_vector_cpu)

get_target_property(_mq_vector_output_dir _mq_vector LIBRARY_OUTPUT_DIRECTORY)
if(NOT _mq_vector_output_dir)
  set(_mq_vector_output_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

if(ENABLE_CUDA)
  pybind11_add_module(_mq_vector_gpu MODULE OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}"
                                                        ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector_gpu.cu)

  target_include_directories(_mq_vector_gpu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
  target_compile_definitions(_mq_vector_gpu PUBLIC ENABLE_GPU)
  set_target_properties(_mq_vector_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  force_at_least_cxx17_workaround(_mq_vector_gpu)
  target_link_libraries(_mq_vector_gpu PUBLIC mqsim_vector_gpu)
  get_target_property(_mq_vector_gpu_output_dir _mq_vector_gpu LIBRARY_OUTPUT_DIRECTORY)
  if(NOT _mq_vector_gpu_output_dir)
    set(_mq_vector_gpu_output_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endif()
endif()

add_subdirectory(include)

if(WIN32)
  get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
  file(
    GLOB
    LIB_LIST
    ${CXX_DIR}/libstdc++-6.dll
    ${CXX_DIR}/libwinpthread-1.dll
    ${CXX_DIR}/libssp-0.dll
    ${CXX_DIR}/libgcc_s_*-1.dll
    ${CXX_DIR}/libgomp-1.dll)
  foreach(win_dep_lib ${LIB_LIST})
    if(EXISTS "${win_dep_lib}")
      message("COPYING ${win_dep_lib}")
      file(COPY ${win_dep_lib} DESTINATION ${_mq_vector_output_dir} FOLLOW_SYMLINK_CHAIN)
    endif()
  endforeach()
endif()
