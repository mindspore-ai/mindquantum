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

#ifndef TO_SYMENGINE_HPP
#define TO_SYMENGINE_HPP

#include <type_traits>
#include <utility>

#include "experimental/core/traits.hpp"
#include "experimental/ops/parametric/concepts.hpp"
#include "experimental/ops/parametric/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::ops::parametric {
#if MQ_HAS_CONCEPTS
template <typename T>
auto to_symengine(T&& t) {
    using type = std::remove_cvref_t<T>;
    if constexpr (std::integral<type>) {
        return SymEngine::integer(t);
    } else if constexpr (std::floating_point<type>) {
        return SymEngine::number(t);
    } else if constexpr (traits::is_std_complex_v<type>) {
        return SymEngine::complex_double(std::forward<T>(t));
    } else {
        return SymEngine::expand(std::forward<T>(t));
    }
}
#else
template <typename T>
auto to_symengine(T&& t) {
    using type = std::remove_cvref_t<T>;
    if constexpr (std::is_integral_v<type>) {
        return SymEngine::integer(t);
    } else if constexpr (std::is_floating_point_v<type>) {
        return SymEngine::number(t);
    } else if constexpr (traits::is_std_complex_v<type>) {
        return SymEngine::complex_double(std::forward<T>(t));
    } else {
        return SymEngine::expand(std::forward<T>(t));
    }
}
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::ops::parametric

#endif /* TO_SYMENGINE_HPP */
