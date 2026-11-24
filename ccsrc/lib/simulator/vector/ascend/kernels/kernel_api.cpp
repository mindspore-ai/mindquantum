// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#include "simulator/vector/ascend/kernel_api.h"

extern "C" __attribute__((visibility("default"))) const mindquantum::sim::ascend::KernelApi* mq_ascend_kernels_v1() {
    namespace ascend = mindquantum::sim::ascend;
    static const ascend::KernelApi api{
        ascend::LaunchTile,
        ascend::LaunchReset,
        ascend::LaunchDense,
        ascend::LaunchAdjointTile,
        ascend::LaunchDenseAdjoint,
        ascend::LaunchObservable,
        ascend::LaunchCompileDiagonal,
        ascend::LaunchInnerProduct,
        ascend::LaunchCollapse,
        ascend::LaunchScale,
        ascend::LaunchDensity,
        ascend::LaunchSamplingTree,
        ascend::LaunchSample,
    };
    return &api;
}
