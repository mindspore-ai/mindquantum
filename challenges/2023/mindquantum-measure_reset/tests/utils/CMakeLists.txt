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

add_library(mindquantum_catch2_main STATIC ${CMAKE_CURRENT_LIST_DIR}/catch2_main.cpp)
target_link_libraries(mindquantum_catch2_main PUBLIC Catch2::Catch2 include_lib
                                                     $<$<BOOL:${ENABLE_LOGGING}>:mindquantum::spdlog>)
target_compile_definitions(mindquantum_catch2_main PUBLIC "$<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING>")

# ==============================================================================

add_library(mindquantum_catch2_utils INTERFACE)
target_include_directories(mindquantum_catch2_utils INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# ==============================================================================
