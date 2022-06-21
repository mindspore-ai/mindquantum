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

set(VER 0.5.1)

if(ENABLE_GITEE)
  set(GIT_URL "https://gitee.com/mirrors/ProjectQ/repository/archive/v${VER}.tar.gz")
  set(MD5 "0da22cba1e2d693b32473b2eaf7c29d2")
else()
  set(GIT_URL "https://github.com/ProjectQ-Framework/ProjectQ/archive/v${VER}.tar.gz")
  set(MD5 "13430199c253284df8b3d840f11d3560")
endif()

set(PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/patch/001-projectq.patch
    ${CMAKE_CURRENT_LIST_DIR}/patch/002-better-aligned-allocator.patch
    ${CMAKE_CURRENT_LIST_DIR}/patch/003-projectq-namespace.patch)

mindquantum_add_pkg(
  projectq
  VER ${VER}
  MD5 ${MD5}
  URL ${GIT_URL}
  ONLY_COPY_DIRS projectq/backends/_sim/_cppkernels
  PATCHES ${PATCHES}
  FORCE_LOCAL_PKG
  TARGET_ALIAS mindquantum::projectq projectq::projectq)
target_compile_features(projectq::projectq INTERFACE cxx_std_14)

# ==============================================================================

add_library(mq_projectq INTERFACE)
target_sources(
  mq_projectq
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/projectq.h>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/projectq_utils.h>
            $<INSTALL_INTERFACE:${MQ_INSTALL_3RDPARTYDIR}/projectq/projectq.h>
            $<INSTALL_INTERFACE:${MQ_INSTALL_3RDPARTYDIR}/projectq/projectq_utils.h>)

target_compile_definitions(mq_projectq INTERFACE INTRIN)
target_include_directories(mq_projectq INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                                 $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}/projectq>)
target_link_libraries(mq_projectq INTERFACE mindquantum::projectq intrin_flag_CXX)

# ------------------------------------------------------------------------------

install(FILES ${CMAKE_CURRENT_LIST_DIR}/projectq.h ${CMAKE_CURRENT_LIST_DIR}/projectq_utils.h
        DESTINATION ${MQ_INSTALL_INCLUDEDIR}/projectq)

append_to_property(mq_install_targets GLOBAL mq_projectq)

# ==============================================================================

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
