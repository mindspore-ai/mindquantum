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
if(ENABLE_CXX_EXPERIMENTAL)
  set(VER 0.7.3)

  if(ENABLE_GITEE)
    set(GIT_URL "https://gitee.com/mirrors/ProjectQ/repository/archive/v${VER}.tar.gz")
    set(MD5 "b78f79c02686079bddf0d4e83517bdf4")
  else()
    set(GIT_URL "https://github.com/ProjectQ-Framework/ProjectQ/archive/v${VER}.tar.gz")
    set(MD5 "83c4407e447ff79344d86e8560a49907")
  endif()

  set(PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/projectq-exp-simulator.patch001)

  mindquantum_add_pkg(
    projectq_experimental
    VER ${VER}
    MD5 ${MD5}
    URL ${GIT_URL}
    PATCHES ${PATCHES}
    ONLY_COPY_DIRS projectq/backends/_sim/_cppkernels
    FORCE_LOCAL_PKG
    TARGET_ALIAS mindquantum::projectq_exp_simulator projectq_experimental::projectq_experimental)

  add_library(mq_projectq_exp INTERFACE)
  target_compile_definitions(mq_projectq_exp INTERFACE INTRIN)
  target_link_libraries(mq_projectq_exp INTERFACE mindquantum::projectq_exp_simulator intrin_flag_CXX)
  append_to_property(mq_install_targets GLOBAL mq_projectq_exp)
endif()
