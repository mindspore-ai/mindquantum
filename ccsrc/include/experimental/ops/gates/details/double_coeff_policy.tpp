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

#ifndef DETAILS_DOUBLE_COEFF_POLICY_TPP
#define DETAILS_DOUBLE_COEFF_POLICY_TPP

#include <optional>
#include <string>

#include <boost/range/iterator_range.hpp>

#include "experimental/ops/gates/details/double_coeff_policy.hpp"

// =============================================================================

namespace mindquantum::ops::details {
template <typename float_t>
auto FloatCoeffPolicyBase<float_t>::coeff_from_string(
    const boost::iterator_range<std::string_view::const_iterator>& range) -> std::optional<coeff_t> {
    return std::stod(std::string(std::begin(range), std::end(range)));
}

}  // namespace mindquantum::ops::details

// =============================================================================

#endif /* DETAILS_DOUBLE_COEFF_POLICY_TPP */
