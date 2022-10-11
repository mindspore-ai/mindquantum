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

set(MQSIM_VECTOR_HEAD ${CMAKE_CURRENT_LIST_DIR}/vector_state.hpp ${CMAKE_CURRENT_LIST_DIR}/vector_state.tpp
                      ${CMAKE_CURRENT_LIST_DIR}/blas.hpp)

set(MQSIM_VECTOR_CPU_HEAD ${CMAKE_CURRENT_LIST_DIR}/detail/cpu_vector_policy.hpp)
list(APPEND MQSIM_VECTOR_CPU_HEAD ${MQSIM_VECTOR_HEAD})

# =============================================================================

if(ENABLE_CUDA)
  set(MQSIM_VECTOR_GPU_HEAD ${CMAKE_CURRENT_LIST_DIR}/detail/cpu_vector_policy.hpp)
  list(APPEND MQSIM_VECTOR_GPU_HEAD ${MQSIM_VECTOR_HEAD})
endif()
