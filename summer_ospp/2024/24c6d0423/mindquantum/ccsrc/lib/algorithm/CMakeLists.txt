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

if(ENABLE_CUDA)
  add_library(qaia_sb_gpu STATIC)
  target_link_libraries(qaia_sb_gpu PUBLIC CUDA_mindquantum mq_math cublas)
  set_target_properties(qaia_sb_gpu PROPERTIES CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC}>,Static,Shared>)
  force_at_least_cxx17_workaround(qaia_sb_gpu)
  append_to_property(mq_install_targets GLOBAL qaia_sb_gpu)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/qaia/detail)
