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

#include <arm_sve.h>

#if !defined(__ARM_FEATURE_SVE)
#    error SVE is unavailable
#endif

int main() {
    const auto pg32 = svptrue_b32();
    const auto value32 = svdup_n_f32(1.0F);
    const auto result32 = svcmla_f32_x(pg32, value32, value32, value32, 0);
    const auto pg64 = svptrue_b64();
    const auto value64 = svdup_n_f64(1.0);
    const auto result64 = svcmla_f64_x(pg64, value64, value64, value64, 0);
    return static_cast<int>(svlastb_f32(pg32, result32) + svlastb_f64(pg64, result64));
}
