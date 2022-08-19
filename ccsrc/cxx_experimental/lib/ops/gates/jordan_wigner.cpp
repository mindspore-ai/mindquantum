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

#include "ops/gates/jordan_wigner.hpp"

#include <cstdint>

#include "ops/gates/fermion_operator_parameter_resolver.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/terms_operator.hpp"

namespace mindquantum::ops::transform {
auto jordan_wigner(const FermionOperatorPR& ops) -> QubitOperatorPR {
    auto transf_op = QubitOperatorPR();
    for (const auto& [term, coeff] : ops.get_terms()) {
        auto transformed_term = QubitOperatorPR(terms_t{}, coeff);
        for (const auto& [idx, value] : term) {
            auto coefficient_1 = QubitOperatorPR::coefficient_t(0.5);
            auto coefficient_2 = QubitOperatorPR::coefficient_t(std::complex<double>(0, -0.5));
            if (value == TermValue::a) {
                coefficient_2 *= -1;
            }
            terms_t t1, t2;
            t1.emplace_back(idx, TermValue::X);
            t2.emplace_back(idx, TermValue::Y);
            for (auto i = 0; i < idx; i++) {
                t1.emplace_back(i, TermValue::Z);
                t2.emplace_back(i, TermValue::Z);
            }
            transformed_term *= (QubitOperatorPR(t1, coefficient_1) + QubitOperatorPR(t2, coefficient_2));
        }
        transf_op += transformed_term;
    }
    return transf_op;
}
}  // namespace mindquantum::ops::transform
