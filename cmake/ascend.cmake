# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

# lint_cmake: -whitespace/indent

include_guard(GLOBAL)
if(NOT ENABLE_ASCEND)
  return()
endif()

set(CMAKE_ASC_ARCHITECTURES
    ""
    CACHE STRING "Ascend compiler architecture, e.g. dav-2201 or dav-3510")
set(MQ_ASCEND_SOC
    ""
    CACHE STRING "Optional CANN SoC selector, used instead of CMAKE_ASC_ARCHITECTURES")
set(ASCEND_CANN_PACKAGE_PATH
    "$ENV{ASCEND_HOME_PATH}"
    CACHE PATH "CANN installation root")
foreach(_selector CMAKE_ASC_ARCHITECTURES MQ_ASCEND_SOC)
  if(NOT "${${_selector}}" STREQUAL "" AND NOT "${${_selector}}" MATCHES "^[A-Za-z0-9_.-]+$")
    message(FATAL_ERROR "${_selector} must be a single compiler target identifier (letters, digits, '.', '_' or '-')")
  endif()
endforeach()
if(CMAKE_ASC_ARCHITECTURES AND MQ_ASCEND_SOC)
  message(FATAL_ERROR "Select CMAKE_ASC_ARCHITECTURES or MQ_ASCEND_SOC, not both")
elseif(NOT CMAKE_ASC_ARCHITECTURES AND NOT MQ_ASCEND_SOC)
  message(FATAL_ERROR "ENABLE_ASCEND requires a compiler target: set CMAKE_ASC_ARCHITECTURES, e.g. dav-3510 for 950")
endif()
# The API probe and both host backends use the same compile-time target selection.
if(CMAKE_ASC_ARCHITECTURES)
  set(MQ_ASCEND_TARGET_DEFINITION "MQ_ASCEND_ARCHITECTURE=\"${CMAKE_ASC_ARCHITECTURES}\"")
else()
  set(MQ_ASCEND_TARGET_DEFINITION "MQ_ASCEND_SOC=\"${MQ_ASCEND_SOC}\"")
endif()
# The installed CANN compiler owns the supported-target list, not MindQuantum. ASCConfig itself reads ASCEND_HOME_PATH.
# Do not combine headers/libraries from one installation with the compiler from another.
if(NOT DEFINED ENV{ASCEND_HOME_PATH}
   OR "$ENV{ASCEND_HOME_PATH}" STREQUAL ""
   OR NOT ASCEND_CANN_PACKAGE_PATH)
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
# Use the component APIs; the legacy ascendcl/opapi umbrella libraries are deprecated by CANN.
foreach(_library acl_rt opapi_nn nnopbase tiling_api platform)
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
set(CMAKE_REQUIRED_LIBRARIES "${MQ_ASCEND_ACL_RT}" "${MQ_ASCEND_OPAPI_NN}" "${MQ_ASCEND_NNOPBASE}")
set(CMAKE_REQUIRED_DEFINITIONS "-D${MQ_ASCEND_TARGET_DEFINITION}")
check_cxx_source_compiles(
  "
  #include <acl/acl.h>
  #include <aclnnop/aclnn_matmul.h>
  int main() {
  #ifdef MQ_ASCEND_ARCHITECTURE
    int64_t architecture = 0;
    aclrtGetDeviceInfo(0, ACL_DEV_ATTR_NPU_ARCH, &architecture);
  #else
    int32_t compatible = 0;
    aclrtCheckArchCompatibility(nullptr, &compatible);
  #endif
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
  MQ_ASCEND_HAS_REQUIRED_APIS)
cmake_pop_check_state()
if(NOT MQ_ASCEND_HAS_REQUIRED_APIS)
  message(
    FATAL_ERROR
      "CANN must provide the selected target's compatibility API, graph capture "
      "and repeatable aclnnMatmul executors. Architecture targets require ACL_DEV_ATTR_NPU_ARCH (CANN 9.1+).")
endif()
