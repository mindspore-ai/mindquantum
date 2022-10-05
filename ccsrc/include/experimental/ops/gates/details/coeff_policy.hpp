//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef DETAILS_COEFF_POLICY_HPP
#define DETAILS_COEFF_POLICY_HPP

#include "config/type_traits.hpp"

namespace mindquantum::ops::details {
#if MQ_HAS_CONCEPTS
template <typename coefficient_t>
struct CoeffPolicy;

template <typename coefficient_t>
struct CoeffSubsProxy;
#else
template <typename coefficient_t, typename = void>
struct CoeffPolicy;

template <typename T, typename = void>
struct CoeffSubsProxy;
#endif  // MQ_HAS_CONCEPTS

inline constexpr auto EQ_TOLERANCE = 1.e-8;

template <typename coefficient_t>
struct CoeffPolicyBase {
    using coeff_policy_real_t = CoeffPolicy<traits::to_real_type_t<coefficient_t>>;

    // Substitute values (if at all supported)
    static auto subs(coefficient_t& coeff, const CoeffSubsProxy<coefficient_t>& subs_params) {
        subs_params.apply(coeff);
    }
};
}  // namespace mindquantum::ops::details

#endif /* DETAILS_COEFF_POLICY_HPP */
