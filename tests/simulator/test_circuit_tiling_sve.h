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

#ifndef TESTS_SIMULATOR_TEST_CIRCUIT_TILING_SVE_H_
#define TESTS_SIMULATOR_TEST_CIRCUIT_TILING_SVE_H_

// Entry point of the SVE-compiled test helper.  Keep Catch2 registrations out
// of the SVE translation unit so non-SVE AArch64 hosts can discover and run
// the test binary without executing SVE instructions.
namespace mindquantum::sim::circuit_tiling::test {
struct SveWindowControlReport {
    bool ok = true;
    const char* failure = "";
};

[[nodiscard]] auto VerifySveExternalControlWindows() -> SveWindowControlReport;
}  // namespace mindquantum::sim::circuit_tiling::test

#endif  // TESTS_SIMULATOR_TEST_CIRCUIT_TILING_SVE_H_
