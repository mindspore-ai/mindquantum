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

add_library(mq_math)
force_at_least_cxx17_workaround(mq_math)
append_to_property(mq_install_targets GLOBAL mq_math)
target_link_libraries(mq_math PUBLIC ${MQ_OPENMP_TARGET} include_lib cmake_config cxx20_compat mq_base)

# ==============================================================================

add_subdirectory(tensor)
add_subdirectory(pr)
add_subdirectory(operators)
add_subdirectory(longbits)
