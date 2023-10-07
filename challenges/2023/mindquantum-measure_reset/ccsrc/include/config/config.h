/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MQ_CONFIG_CONFIG_HPP
#define MQ_CONFIG_CONFIG_HPP

#include "config/cmake_config.h"
#ifdef __CUDACC__
#    include "config/cuda20_config.h"
#else
#    include "config/cxx20_config.h"
#endif  // __CUDACC__
#include "config/details/clang_version.h"
#include "config/details/cxx20_compatibility.h"
#include "config/details/macros.h"
#include "config/type_traits.h"

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

#ifndef MQ_IS_CLANG_VERSION_LESS
#    define MQ_IS_CLANG_VERSION_LESS(major, minor)                                                                     \
        (defined __clang__) && (MQ_CLANG_MAJOR < (major)) && (MQ_CLANG_MINOR < (minor))
#    define MQ_IS_CLANG_VERSION_LESS_EQUAL(major, minor)                                                               \
        (defined __clang__) && (MQ_CLANG_MAJOR <= (major)) && (MQ_CLANG_MINOR <= (minor))
#endif  // MQ_IS_CLANG_VERSION_LESS

/*!
 * \def MQ_IS_CLANG_VERSION_LESS(major, minor)
 * True if the compiler is Clang and if its version is strictly less than <major>.<minor>
 */
/*!
 * \def MQ_IS_CLANG_VERSION_LESS_EQUAL(major, minor)
 * True if the compiler is Clang and if its version is less than or equal to <major>.<minor>
 */

// -------------------------------------

#if !defined(MQ_CONFIG_NO_COUNTER) && !defined(MQ_CONFIG_COUNTER)
#    define MQ_CONFIG_COUNTER
#endif  // !MQ_CONFIG_NO_COUNTER && !MQ_CONFIG_COUNTER

#define MQ_UNIQUE_NAME_LINE2_(name, line) name##line
#define MQ_UNIQUE_NAME_LINE_(name, line)  MQ_UNIQUE_NAME_LINE2_(name, line)
#ifdef MQ_CONFIG_COUNTER
#    define MQ_UNIQUE_NAME(name) MQ_UNIQUE_NAME_LINE_(name, __COUNTER__)
#else
#    define MQ_UNIQUE_NAME(name) MQ_UNIQUE_NAME_LINE_(name, __LINE__)
#endif

/*!
 * \def MQ_UNIQUE_NAME(name)
 * Define a unique and valid C++ identifier.
 *
 * This is either based on \c __COUNTER__ if supported or in \c __LINE__ otherwise.
 */

// =============================================================================

#endif /* MQ_CONFIG_CONFIG_HPP */
