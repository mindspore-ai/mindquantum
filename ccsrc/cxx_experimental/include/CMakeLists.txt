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

add_library(include_lib INTERFACE)
target_include_directories(include_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                                 $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}/experimental>)
append_to_property(mq_install_targets GLOBAL include_lib)

target_link_libraries(mindquantum_cxx PUBLIC include_lib)

# ------------------------------------------------------------------------------

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cengines
            ${CMAKE_CURRENT_LIST_DIR}/core
            ${CMAKE_CURRENT_LIST_DIR}/decompositions
            ${CMAKE_CURRENT_LIST_DIR}/details
            ${CMAKE_CURRENT_LIST_DIR}/mapping
            ${CMAKE_CURRENT_LIST_DIR}/ops
            ${CMAKE_CURRENT_LIST_DIR}/simulator
  DESTINATION ${MQ_INSTALL_INCLUDEDIR}/experimental)

# ==============================================================================
