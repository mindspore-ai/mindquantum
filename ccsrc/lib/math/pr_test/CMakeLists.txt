# ==============================================================================
#
# Copyright 2023 <Huawei Technologies Co., Ltd>
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

add_executable(pr_test)
target_link_libraries(pr_test PUBLIC mq_math)
force_at_least_cxx17_workaround(pr_test)
append_to_property(mq_install_targets GLOBAL pr_test)
target_sources(pr_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pr_test.cpp)
