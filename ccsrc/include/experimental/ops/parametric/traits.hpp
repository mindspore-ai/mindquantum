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

#ifndef PARAMETRIC_TRAITS_HPP
#define PARAMETRIC_TRAITS_HPP

#include <utility>

#include "experimental/ops/parametric/config.hpp"

namespace mindquantum::traits {
template <typename ref_kind_t, typename... kinds_t>
constexpr bool kind_match(ref_kind_t&& ref_kind, kinds_t&&... kinds)
#if MQ_HAS_CONCEPTS
    requires(sizeof...(kinds_t) > 0)
#endif  // MQ_HAS_CONCEPTS
{
#if !MQ_HAS_CONCEPTS
    static_assert(sizeof...(kinds_t) > 0);
#endif  // MQ_HAS_CONCEPTS
    return ((ref_kind == kinds) || ...);
}
}  // namespace mindquantum::traits

#endif /* PARAMETRIC_TRAITS_HPP */
