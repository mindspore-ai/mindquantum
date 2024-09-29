# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# lint_cmake: -whitespace/indent

add_library(mq_base STATIC)
set_target_properties(mq_base PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_features(mq_base PUBLIC $<IF:$<COMPILE_FEATURES:cxx_std_20>,cxx_std_20,cxx_std_17>)
append_to_property(mq_install_targets GLOBAL mq_base)
force_at_least_cxx17_workaround(mq_base)
target_link_libraries(
  mq_base
  PUBLIC ${MQ_OPENMP_TARGET}
         cxx20_compat
         cmake_config
         mq_math
         mindquantum::fmt
         mindquantum::json
         mindquantum_setup
         $<$<BOOL:${ENABLE_LOGGING}>:mindquantum::spdlog>)

# ==============================================================================

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)

# ==============================================================================
