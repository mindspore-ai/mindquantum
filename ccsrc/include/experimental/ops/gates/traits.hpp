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

#ifndef OPS_GATES_TRAITS_HPP
#define OPS_GATES_TRAITS_HPP

#include <complex>

#include "experimental/core/traits.hpp"

namespace mindquantum::traits {
// TODO(dnguyen): Remove this if not needed anymore.
template <typename T>
inline constexpr auto is_termsop_number = std::is_floating_point_v<T> || is_std_complex_v<T>;

}  // namespace mindquantum::traits

#endif /* OPS_GATES_TRAITS_HPP */
