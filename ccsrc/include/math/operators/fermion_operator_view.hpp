//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_
#define MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_

#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "math/pr/parameter_resolver.hpp"
namespace operators::fermion {
enum class TermValue : uint64_t {
    //! DO NOT CHANGE VALUE.
    I = 0,                                       // 000
    A = 1,                                       // 001
    Ad = 2,                                      // 010
    AdA = 3,                                     // 011
    AAd = 6,                                     // 110
    nll = std::numeric_limits<uint64_t>::max(),  // 11111...
};

using fermion_product_t = std::map<TermValue, std::map<TermValue, TermValue>>;
const fermion_product_t fermion_product = {
    {
        TermValue::I,
        {
            {TermValue::I, TermValue::I},
            {TermValue::A, TermValue::A},
            {TermValue::Ad, TermValue::Ad},
            {TermValue::AdA, TermValue::AdA},
            {TermValue::AAd, TermValue::AAd},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::A,
        {
            {TermValue::I, TermValue::A},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::AAd},
            {TermValue::AdA, TermValue::A},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::Ad,
        {
            {TermValue::I, TermValue::Ad},
            {TermValue::A, TermValue::AdA},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::Ad},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::AdA,
        {
            {TermValue::I, TermValue::AdA},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::Ad},
            {TermValue::AdA, TermValue::AdA},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::AAd,
        {
            {TermValue::I, TermValue::AAd},
            {TermValue::A, TermValue::A},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::AAd},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::nll,
        {
            {TermValue::I, TermValue::nll},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
};

// -----------------------------------------------------------------------------

struct SingleFermionStr {
    using key_t = std::vector<uint64_t>;
    using value_t = parameter::ParameterResolver;
    using fermion_t = std::pair<key_t, value_t>;
};

}  // namespace operators::fermion

#endif /* MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_ */
