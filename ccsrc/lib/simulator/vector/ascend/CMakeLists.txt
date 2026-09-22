# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

set(MQ_ASCEND_SOC "" CACHE STRING "Exact Ascend C target, e.g. Ascend910B2")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN installation root")
if(NOT MQ_ASCEND_SOC MATCHES "^Ascend910B[1-4]$")
  message(FATAL_ERROR "ENABLE_ASCEND requires an explicit MQ_ASCEND_SOC=Ascend910B[1-4]")
endif()
find_path(MQ_ASCEND_INCLUDE acl/acl.h HINTS "${ASCEND_CANN_PACKAGE_PATH}/include" REQUIRED)
find_library(MQ_ASCEND_LIBRARY ascendcl HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
find_library(MQ_ASCEND_TILING tiling_api HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
find_library(MQ_ASCEND_PLATFORM platform HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
# ASCConfig changes host compiler flags. Keep its scope out of host targets.
add_subdirectory(kernels)

add_library(mqsim_vector_npu STATIC program.cpp runtime.cpp observable.cpp)
# Compilation validates finite coefficients; the project's fast-math flag must
# not optimize those checks away. This applies after target usage requirements.
set_source_files_properties(program.cpp runtime.cpp observable.cpp PROPERTIES COMPILE_OPTIONS -fno-fast-math)
target_include_directories(mqsim_vector_npu PUBLIC "${MQ_ASCEND_INCLUDE}")
target_link_libraries(mqsim_vector_npu PUBLIC mq_math PRIVATE mqsim_circuit_tiling mq_ascend_kernels
                     "${MQ_ASCEND_LIBRARY}" "${MQ_ASCEND_TILING}" "${MQ_ASCEND_PLATFORM}")
target_compile_definitions(mqsim_vector_npu PRIVATE MQ_ASCEND_SOC="${MQ_ASCEND_SOC}")
force_at_least_cxx17_workaround(mqsim_vector_npu)
