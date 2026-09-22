# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake")
enable_language(ASC)
ascendc_library(mq_ascend_kernels STATIC statevector.asc dense.asc gradient.asc observable.asc)
target_include_directories(mq_ascend_kernels PRIVATE "${PROJECT_SOURCE_DIR}/ccsrc/include" "${MQ_ASCEND_INCLUDE}")
target_compile_options(mq_ascend_kernels PRIVATE --npu-soc=${MQ_ASCEND_SOC} -O3 -fno-fast-math -ffp-contract=off)
target_compile_definitions(mq_ascend_kernels PRIVATE ASCENDC_DUMP=0)
