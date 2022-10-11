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

pybind11_add_module(_mq_vector MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector.cpp
                    OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_vector PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_vector)
target_link_libraries(_mq_vector PUBLIC mqsim_vector_cpu)

# ------------------------------------------------------------------------------

if(ENABLE_CUDA)
  pybind11_add_module(_mq_vector_gpu MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector_gpu.cu
                      OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

  target_include_directories(_mq_vector_gpu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
  target_compile_definitions(_mq_vector_gpu PUBLIC ENABLE_GPU)
  set_target_properties(_mq_vector_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  force_at_least_cxx17_workaround(_mq_vector_gpu)
  target_link_libraries(_mq_vector_gpu PUBLIC mqsim_vector_gpu)
endif()

# ------------------------------------------------------------------------------

add_subdirectory(include)
