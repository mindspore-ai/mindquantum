# ==============================================================================
#
# Copyright 2026 <Huawei Technologies Co., Ltd>
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

# Arm SVE/FCMA simulator kernels are compiled into dedicated translation units
# and selected at runtime once the host CPU has been probed.  This module
# decides once whether those units can be built and adds them to targets.

include(CheckCXXSourceCompiles)

set(MQ_ARM_SVE_COMPILE_OPTION "-march=armv8-a+sve")
set(MQ_ARM_SVE_AVAILABLE OFF)

if(MQ_ARM_SVE AND AARCH64 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_mq_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${_mq_saved_required_flags} ${MQ_ARM_SVE_COMPILE_OPTION}")
  file(READ "${PROJECT_SOURCE_DIR}/cmake/probes/arm_sve_fcma.cpp" _mq_arm_sve_probe)
  check_cxx_source_compiles("${_mq_arm_sve_probe}" MQ_COMPILER_HAS_ARM_SVE_FCMA)
  set(CMAKE_REQUIRED_FLAGS "${_mq_saved_required_flags}")
  unset(_mq_arm_sve_probe)
  unset(_mq_saved_required_flags)

  if(NOT MQ_COMPILER_HAS_ARM_SVE_FCMA)
    message(FATAL_ERROR "MQ_ARM_SVE=ON, but the C++ compiler cannot build the required SVE/FCMA intrinsics.")
  endif()
  set(MQ_ARM_SVE_AVAILABLE ON)
  message(STATUS "Arm simulator kernels: baseline, SVE/FCMA (runtime dispatched; ${MQ_ARM_SVE_COMPILE_OPTION})")
elseif(AARCH64)
  message(STATUS "Arm simulator kernels: baseline")
endif()

# ~~~
# mq_add_arm_sve_sources(<target> <source> [... <source>])
#
# Adds SVE/FCMA translation units to <target> when they can be built, and
# defines MQ_HAS_ARM_SVE_BACKEND on the target accordingly.
# ~~~
function(mq_add_arm_sve_sources target)
  if(MQ_ARM_SVE_AVAILABLE)
    target_sources(${target} PRIVATE ${ARGN})
    set_source_files_properties(${ARGN} TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS
                                                                                 "${MQ_ARM_SVE_COMPILE_OPTION}")
    target_compile_definitions(${target} PRIVATE MQ_HAS_ARM_SVE_BACKEND=1)
  else()
    target_compile_definitions(${target} PRIVATE MQ_HAS_ARM_SVE_BACKEND=0)
  endif()
endfunction()
