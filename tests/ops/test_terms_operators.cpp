//   Copyright 2021 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <complex>

#include "ops/gates/terms_operator.hpp"
#include "ops/utils.hpp"

// =============================================================================

namespace {
struct A : mindquantum::ops::TermsOperator<A> {
    static std::tuple<std::vector<term_t>, coefficient_t> simplify_(const std::vector<term_t>& terms,
                                                                    coefficient_t coeff) {
        return {};
    }
};
}  // namespace
// =============================================================================

TEST_CASE("Test terms operators", "[terms][ops]") {
    using namespace std::literals::complex_literals;

    ::A op, other;
    op += other;
    op += 2.0;
    op += 2.i;
    op + other;

    op -= other;
    op -= 2.0;
    op -= 2.i;
    op - other;

    op *= other;

    op.pow(10);

    CHECK(op == other);
}

// =============================================================================
