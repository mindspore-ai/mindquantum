# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

mq_ascend_kernel_library(
  mq_ascend_kernels
  _mq_vector_npu
  statevector.asc
  dense.asc
  gradient.asc
  observable.asc
  kernel_api.cpp)
