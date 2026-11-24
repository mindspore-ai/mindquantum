// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#include "algorithm/qaia/ascend/kernel_launch.h"

extern "C" __attribute__((visibility("default"))) const mindquantum::algorithm::qaia::ascend::KernelApi*
mq_ascend_kernels_v1() {
    static const mindquantum::algorithm::qaia::ascend::KernelApi api{
        mindquantum::algorithm::qaia::ascend::LaunchSbStep};
    return &api;
}
