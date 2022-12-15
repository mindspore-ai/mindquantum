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

add_test_executable(test_common_type LIBS include_lib cmake_config mindquantum::json mindquantum::fmt)
if(NOT MSVC OR NOT COMPILER_IS_CLANG_CL)
  add_test_executable(test_constexpr_type_name LIBS include_lib cmake_config mindquantum::fmt)
endif()
