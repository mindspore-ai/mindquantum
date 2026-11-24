# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

# lint_cmake: -whitespace/indent

add_library(mqsim_vector_npu STATIC program.cpp runtime.cpp observable.cpp)
# Compilation validates finite coefficients; the project's fast-math flag must not optimize those checks away. This
# applies after target usage requirements.
set_source_files_properties(program.cpp runtime.cpp observable.cpp PROPERTIES COMPILE_OPTIONS -fno-fast-math)
target_include_directories(mqsim_vector_npu PUBLIC "${MQ_ASCEND_INCLUDE}")
target_link_libraries(
  mqsim_vector_npu
  PUBLIC mq_math
  PRIVATE mqsim_circuit_tiling "${MQ_ASCEND_ACL_RT}" "${MQ_ASCEND_TILING_API}" "${MQ_ASCEND_PLATFORM}"
          "${MQ_ASCEND_REGISTER}" ${CMAKE_DL_LIBS})
target_compile_definitions(mqsim_vector_npu PRIVATE "${MQ_ASCEND_TARGET_DEFINITION}")
force_at_least_cxx17_workaround(mqsim_vector_npu)
