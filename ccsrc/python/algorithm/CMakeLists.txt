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
  pybind11_add_module(_qaia_sb MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_qaia_sb.cu
                      OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")
  force_at_least_cxx17_workaround(_qaia_sb)
  target_link_libraries(_qaia_sb PUBLIC mq_python_core qaia_sb_gpu)
  set_target_properties(_qaia_sb PROPERTIES CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC}>,Static,Shared>)
endif()

if(ENABLE_ASCEND)
  pybind11_add_module(_qaia_sb_ascend MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_qaia_sb_ascend.cpp
                      OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")
  force_at_least_cxx17_workaround(_qaia_sb_ascend)
  # Source options follow transitive fast-math flags: isfinite must remain real validation.
  set_source_files_properties(lib/_qaia_sb_ascend.cpp PROPERTIES COMPILE_OPTIONS -fno-fast-math)
  target_include_directories(_qaia_sb_ascend PRIVATE "${MQ_ASCEND_INCLUDE}" "${MQ_ASCEND_ACLNN_INCLUDE}")
  target_compile_definitions(_qaia_sb_ascend PRIVATE MQ_ASCEND_SOC="${MQ_ASCEND_SOC}")
  target_link_libraries(_qaia_sb_ascend PRIVATE mq_python_core qaia_sb_npu_kernels "${MQ_ASCEND_ASCENDCL}"
                                                "${MQ_ASCEND_OPAPI}" "${MQ_ASCEND_NNOPBASE}")
endif()
