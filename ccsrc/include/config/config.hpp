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

#ifndef MQ_CORE_CONFIG_HPP
#define MQ_CORE_CONFIG_HPP

#include "config/type_traits.hpp"

// =============================================================================

#ifdef __has_cpp_attribute
#    if __has_cpp_attribute(nodiscard)
#        define MQ_NODISCARD [[nodiscard]]
#    endif  // __has_cpp_attribute(nodiscard)
#endif      // __has_cpp_attribute

#ifndef MQ_NODISCARD
#    define MQ_NODISCARD
#endif  // MQ_NODISCARD

/*!
 * \def MQ_NODISCARD
 * Add the [[nodiscard]] attribute to a function if supported by the compiler
 */

// -------------------------------------

#ifndef _MSC_VER
#    define MQ_ALIGN(x) __attribute__((aligned(x)))
#else
#    define MQ_ALIGN(x)
#endif  // !_MSC_VER

/*!
 * \def MQ_ALIGN(x)
 * Align a C++ struct/class to a specific size.
 */

// =============================================================================

#endif /* MQ_CORE_CONFIG_HPP */
