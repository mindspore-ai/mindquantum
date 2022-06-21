//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#ifndef VISITOR_HPP
#define VISITOR_HPP

namespace mindquantum {
//! Helper class to generate visitors for std::variant
/*!
 * This performs some "black magick" in order to allow users to create a
 * visitor class consisting only of lambdas at the call site.
 *
 * It uses four key language features:
 * - variadic templates (C++11)
 * - aggregate initialization (C++11)
 * - pack expansion of using directice (C++17)
 * - custom template argument deduction rules (C++17)
 *
 * For example, the following would create an instance of a \c overload
 * class with two overload for operator(): one taking an \c int and one
 * taking a \c double.
 *
 * \code
 *     overload{
 *         [] (int i) {  std::cout << "int: " << i; },
 *         [] (double d::cout << "double: " << d; },
 *     }
 * \endcode
 */
template <class... Ts>
struct overload : Ts... {
    using Ts::operator()...;
};

#if !MQ_HAS_IMPLICIT_TEMPLATE_DEDUCTION_GUIDES
//! Template deduction guide for the overload struct
/*!
 * \note Not needed anymore for C++20
 */
template <class... Ts>
overload(Ts...) -> overload<Ts...>;
#endif /* MQ_HAS_IMPLICIT_TEMPLATE_DEDUCTION_GUIDES */
}  // namespace mindquantum

#endif /* VISITOR_HPP */
