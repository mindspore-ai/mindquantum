# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake")
enable_language(ASC)
ascendc_library(qaia_sb_npu_kernels STATIC sb_kernel.asc)
target_include_directories(qaia_sb_npu_kernels PUBLIC "${PROJECT_SOURCE_DIR}/ccsrc/include" "${MQ_ASCEND_INCLUDE}")
target_compile_options(qaia_sb_npu_kernels PRIVATE --npu-soc=${MQ_ASCEND_SOC} -O3 -fno-fast-math -ffp-contract=off)
target_compile_definitions(qaia_sb_npu_kernels PRIVATE ASCENDC_DUMP=0)
