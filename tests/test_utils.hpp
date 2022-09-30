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

#ifndef TESTS_UTILS_HPP
#define TESTS_UTILS_HPP

#include <ostream>

// =============================================================================
// NB: Need to have those defined *before* including Catch2

namespace internal::traits {
template <typename T, typename = void>
struct to_string : std::false_type {};
template <typename T>
struct to_string<T, std::void_t<decltype(std::declval<const T&>().to_string())>> : std::true_type {};

}  // namespace internal::traits

// -------------------------------------

template <typename T>
std::enable_if_t<internal::traits::to_string<T>::value, std::ostream&> operator<<(std::ostream& out, const T& object) {
    return out << object.to_string();
}

// -----------------------------------------------------------------------------

#include <catch2/catch_all.hpp>

// =============================================================================

#endif /* TESTS_UTILS_HPP */
