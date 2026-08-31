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

#ifndef INCLUDE_SIMULATOR_CPU_FEATURES_H_
#define INCLUDE_SIMULATOR_CPU_FEATURES_H_

#include <cstdint>

namespace mindquantum::sim::cpu {
enum class CpuFeature : std::uint64_t {
    ArmFcma = static_cast<std::uint64_t>(1) << 0,
    ArmSve = static_cast<std::uint64_t>(1) << 1,
};

class CpuFeatureSet {
 public:
    constexpr CpuFeatureSet() noexcept = default;
    constexpr explicit CpuFeatureSet(CpuFeature feature) noexcept : bits_(static_cast<std::uint64_t>(feature)) {
    }

    [[nodiscard]] constexpr auto Contains(CpuFeature feature) const noexcept -> bool {
        return (bits_ & static_cast<std::uint64_t>(feature)) != 0;
    }

    constexpr auto Add(CpuFeature feature) noexcept -> CpuFeatureSet& {
        bits_ |= static_cast<std::uint64_t>(feature);
        return *this;
    }

 private:
    std::uint64_t bits_ = 0;
};

[[nodiscard]] auto HostCpuFeatures() noexcept -> const CpuFeatureSet&;

// The Arm SVE simulator kernels rely on FCMLA complex arithmetic, so both
// features must be present before they may be selected.
[[nodiscard]] inline auto HostHasArmSveFcma() noexcept -> bool {
    const auto& features = HostCpuFeatures();
    return features.Contains(CpuFeature::ArmFcma) && features.Contains(CpuFeature::ArmSve);
}

// The Arm SVE vector length of this process in bytes, or 0 without SVE.
[[nodiscard]] auto ArmSveVectorLengthBytes() noexcept -> unsigned;
}  // namespace mindquantum::sim::cpu

#endif  // INCLUDE_SIMULATOR_CPU_FEATURES_H_
