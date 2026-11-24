// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_API_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_API_H_

#include "simulator/vector/ascend/kernel_launch.h"
#include "simulator/vector/ascend/observable.h"

namespace mindquantum::sim::ascend {
// Private to this build: no C++ objects, allocations or exceptions cross the
// kernel-library boundary. Signatures come from the existing launch contract.
struct KernelApi {
    decltype(&LaunchTile) tile;
    decltype(&LaunchReset) reset;
    decltype(&LaunchDense) dense;
    decltype(&LaunchAdjointTile) adjoint_tile;
    decltype(&LaunchDenseAdjoint) dense_adjoint;
    decltype(&LaunchObservable) observable;
    decltype(&LaunchCompileDiagonal) compile_diagonal;
    decltype(&LaunchInnerProduct) inner_product;
    decltype(&LaunchCollapse) collapse;
    decltype(&LaunchScale) scale;
    decltype(&LaunchDensity) density;
    decltype(&LaunchSamplingTree) sampling_tree;
    decltype(&LaunchSample) sample;
};
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_API_H_
