# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

# lint_cmake: -whitespace/indent

include_guard(GLOBAL)
if(NOT ENABLE_ASCEND)
  return()
endif()

set(MQ_ASCEND_SOC
    ""
    CACHE STRING "Exact Ascend C target, e.g. Ascend910B2")
set(ASCEND_CANN_PACKAGE_PATH
    "$ENV{ASCEND_HOME_PATH}"
    CACHE PATH "CANN installation root")
if(NOT MQ_ASCEND_SOC MATCHES "^Ascend910B[1-4]$")
  message(FATAL_ERROR "ENABLE_ASCEND requires an explicit MQ_ASCEND_SOC=Ascend910B[1-4]")
endif()
# ASCConfig itself reads ASCEND_HOME_PATH. Do not combine headers/libraries from one installation with the compiler from
# another.
if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "" OR NOT ASCEND_CANN_PACKAGE_PATH)
  message(FATAL_ERROR "Source CANN set_env.sh; ASCEND_HOME_PATH and ASCEND_CANN_PACKAGE_PATH must be nonempty")
endif()
get_filename_component(_cann_root "${ASCEND_CANN_PACKAGE_PATH}" REALPATH)
get_filename_component(_cann_env "$ENV{ASCEND_HOME_PATH}" REALPATH)
if(NOT _cann_root STREQUAL _cann_env)
  message(FATAL_ERROR "Source CANN set_env.sh; ASCEND_CANN_PACKAGE_PATH must match ASCEND_HOME_PATH")
endif()
# ASC is a CMake language: its compiler and generated rules are cached too. Resetting only find_path/find_library
# results would still mix toolchains. Like changing the host compiler, changing CANN requires a fresh build directory.
if(DEFINED MQ_ASCEND_CANN_ROOT AND NOT MQ_ASCEND_CANN_ROOT STREQUAL _cann_root)
  message(
    FATAL_ERROR "CANN installation changed from ${MQ_ASCEND_CANN_ROOT} to ${_cann_root}; use a new build directory")
endif()
set(MQ_ASCEND_CANN_ROOT
    "${_cann_root}"
    CACHE INTERNAL "CANN installation used by this build directory")
find_path(
  MQ_ASCEND_INCLUDE acl/acl.h
  HINTS "${_cann_root}/include"
  NO_DEFAULT_PATH REQUIRED)
find_path(
  MQ_ASCEND_ACLNN_INCLUDE aclnnop/aclnn_matmul.h
  HINTS "${_cann_root}/include"
  NO_DEFAULT_PATH REQUIRED)
foreach(_library ascendcl opapi nnopbase tiling_api platform)
  string(TOUPPER "${_library}" _name)
  find_library(
    MQ_ASCEND_${_name} ${_library}
    HINTS "${_cann_root}/lib64"
    NO_DEFAULT_PATH REQUIRED)
endforeach()

# Fail during configuration, not while building or importing a Python module.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES "${MQ_ASCEND_INCLUDE}" "${MQ_ASCEND_ACLNN_INCLUDE}")
set(CMAKE_REQUIRED_LIBRARIES "${MQ_ASCEND_ASCENDCL}" "${MQ_ASCEND_OPAPI}" "${MQ_ASCEND_NNOPBASE}")
check_cxx_source_compiles(
  "
  #include <acl/acl.h>
  #include <aclnnop/aclnn_matmul.h>
  int main() {
    aclmdlRI graph = nullptr;
    aclmdlRICaptureBegin(nullptr, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    aclmdlRICaptureEnd(nullptr, &graph);
    aclmdlRIExecuteAsync(graph, nullptr);
    aclmdlRIDestroy(graph);
    aclSetAclOpExecutorRepeatable(nullptr);
    aclDestroyAclOpExecutor(nullptr);
    aclnnMatmulGetWorkspaceSize(nullptr, nullptr, nullptr, 0, nullptr, nullptr);
    return aclnnMatmul(nullptr, 0, nullptr, nullptr);
  }
  "
  MQ_ASCEND_HAS_GRAPH_MATMUL)
cmake_pop_check_state()
if(NOT MQ_ASCEND_HAS_GRAPH_MATMUL)
  message(FATAL_ERROR "CANN must provide aclmdlRI graph capture and repeatable aclnnMatmul executors")
endif()
