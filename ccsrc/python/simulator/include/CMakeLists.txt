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

add_library(bind_lib INTERFACE)
target_include_directories(bind_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                              $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}/>)
target_link_libraries(_mq_vector PUBLIC bind_lib)
append_to_property(mq_install_targets GLOBAL bind_lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/python DESTINATION ${MQ_INSTALL_INCLUDEDIR})

if(ENABLE_GPU)
  add_library(bind_lib_gpu INTERFACE)
  target_include_directories(bind_lib_gpu INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                                    $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}/>)
  target_link_libraries(_mq_vector_gpu PUBLIC bind_lib_gpu)
  append_to_property(mq_install_targets GLOBAL bind_lib_gpu)
  target_compile_definitions(bind_lib_gpu PUBLIC ENABLE_GPU)
endif()
