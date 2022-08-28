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

#include "ops/gates/terms_operator.hpp"
#include "ops/transform/parity.hpp"
#include "ops/transform/transform_ladder_operator.hpp"
#include "ops/transform/types.hpp"

namespace mindquantum::ops::transform {
// template <typename fermion_t, typename qubit_t>
fermion_t fermion_number_operator(int n_modes, int mode, fermion_t::coefficient_t coeff) {
    fermion_t out{};
    if (mode < 0) {
        for (auto m = 0; m < n_modes; m++) {
            out += fermion_number_operator(n_modes, m, coeff);
        }
    } else {
        out += fermion_t({{mode, TermValue::adg}, {mode, TermValue::a}}, coeff);
    }
    return out;
}
}  // namespace mindquantum::ops::transform
