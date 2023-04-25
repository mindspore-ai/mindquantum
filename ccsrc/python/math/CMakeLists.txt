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
add_library(_math_python_interface INTERFACE)
target_include_directories(_math_python_interface INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                                            $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}>)
append_to_property(mq_install_targets GLOBAL _math_python_interface)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/python DESTINATION ${MQ_INSTALL_INCLUDEDIR})

pybind11_add_module(_math MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/bind_math.cpp OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")
force_at_least_cxx17_workaround(_mq_vector)
target_link_libraries(_math PUBLIC mq_math _math_python_interface)
