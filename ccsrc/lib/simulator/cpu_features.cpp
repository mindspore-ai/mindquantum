/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(__linux__) && defined(__aarch64__)
#    include <sys/auxv.h>
#endif

#include "simulator/cpu_features.h"

namespace mindquantum::sim::cpu {
namespace {
#if defined(__linux__) && defined(__aarch64__)
// Stable Linux AArch64 UAPI values from asm/hwcap.h.  Keeping the ABI values
// here lets a wheel built against an older manylinux sysroot discover features
// supplied by the user's newer kernel.
inline constexpr unsigned long kLinuxHwcapFcma = static_cast<unsigned long>(1) << 14;
inline constexpr unsigned long kLinuxHwcapSve = static_cast<unsigned long>(1) << 22;
#endif

[[nodiscard]] auto DetectHostCpuFeatures() noexcept -> CpuFeatureSet {
    auto features = CpuFeatureSet{};
#if defined(__linux__) && defined(__aarch64__)
    const auto hwcap = getauxval(AT_HWCAP);
    if ((hwcap & kLinuxHwcapFcma) != 0) {
        features.Add(CpuFeature::ArmFcma);
    }
    if ((hwcap & kLinuxHwcapSve) != 0) {
        features.Add(CpuFeature::ArmSve);
    }
#endif
    return features;
}
}  // namespace

auto HostCpuFeatures() noexcept -> const CpuFeatureSet& {
    static const auto features = DetectHostCpuFeatures();
    return features;
}
}  // namespace mindquantum::sim::cpu
