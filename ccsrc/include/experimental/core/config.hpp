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

#ifndef CORE_CONFIG_HPP
#define CORE_CONFIG_HPP

#include <vector>

#include "config/config.hpp"
#include "config/logging.hpp"

#include "experimental/core/types.hpp"

// =============================================================================

#if MQ_HAS_CONCEPTS
#    define MQ_REQUIRES(x) requires(x)
#else
#    define MQ_REQUIRES(x)
#endif  // MQ_HAS_CONCEPTS

/*!
 * \def MQ_REQUIRES
 * Add a C++20 requires() clause to a function if supported by the compiler.
 */

#endif /* CORE_CONFIG_HPP */
