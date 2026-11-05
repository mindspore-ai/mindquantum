// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#ifndef INCLUDE_ALGORITHM_QAIA_ASCEND_KERNEL_LAUNCH_H_
#define INCLUDE_ALGORITHM_QAIA_ASCEND_KERNEL_LAUNCH_H_

#include <cstdint>

#include <acl/acl.h>

namespace mindquantum::algorithm::qaia::ascend {
// FP32 buffers use [batch, n] storage; n must be a positive multiple of 64.
// h is broadcast across replicas; iteration=-1 initializes DSB's sign buffer.
aclError LaunchSbStep(uint32_t blocks, aclrtStream stream, void* x, void* y, void* coupling, void* spin, void* h,
                      void* p, int32_t n, int32_t batch, int32_t iteration, int32_t discrete, float delta, float xi,
                      float dt);
}  // namespace mindquantum::algorithm::qaia::ascend
#endif  // INCLUDE_ALGORITHM_QAIA_ASCEND_KERNEL_LAUNCH_H_
