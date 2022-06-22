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

add_library(mindquantum_cxx STATIC)
set_target_properties(mindquantum_cxx PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(mindquantum_cxx PUBLIC cxx20_compat cmake_config mindquantum_setup mq_projectq_exp
                                             ${MQ_OPENMP_TARGET})
append_to_property(mq_install_targets GLOBAL mindquantum_cxx)

target_compile_features(mindquantum_cxx PUBLIC $<IF:$<COMPILE_FEATURES:cxx_std_20>,cxx_std_20,cxx_std_17>)
force_at_least_cxx17_workaround(mindquantum_cxx)

add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(python)
