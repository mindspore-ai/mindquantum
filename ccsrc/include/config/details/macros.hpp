/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef CORE_DETAILS_MACROS_HPP
#define CORE_DETAILS_MACROS_HPP

// =============================================================================

// NB: _Pragma(...) is a C++11 addition and should be widely supported
#define MQ_DO_PRAGMA(x) _Pragma(#x)

// =============================================================================

#ifdef __clang__
#    define CLANG_DIAG_PRAGMA(x) MQ_DO_PRAGMA(GCC diagnostic x)
#    define CLANG_DIAG_OFF(x)                                                                                          \
        CLANG_DIAG_PRAGMA(push)                                                                                        \
        CLANG_DIAG_PRAGMA(ignored x)
#    define CLANG_DIAG_ON(x) CLANG_DIAG_PRAGMA(pop)
#else
#    define CLANG_DIAG_OFF(x)
#    define CLANG_DIAG_ON(x)
#endif  // __clang__

// =============================================================================

#if (defined __GNUC__) && !(defined __clang__)
#    define GCC_DIAG_PRAGMA(x) MQ_DO_PRAGMA(GCC diagnostic x)
#    define GCC_DIAG_OFF(x)                                                                                            \
        GCC_DIAG_PRAGMA(push)                                                                                          \
        GCC_DIAG_PRAGMA(ignored x)
#    define GCC_DIAG_ON(x) GCC_DIAG_PRAGMA(pop)
#else
#    define GCC_DIAG_OFF(x)
#    define GCC_DIAG_ON(x)
#endif  // __GNUC__

// =============================================================================

#ifdef _MSC_VER
#    define MSVC_DIAG_OFF(x) MQ_DO_PRAGMA(warning(push)) MQ_DO_PRAGMA(warning(disable : warningNumber))
#    define MSVC_DIAG_ON(x)  MQ_DO_PRAGMA(warning(pop))
#else
#    define MSVC_DIAG_OFF(x)
#    define MSVC_DIAG_ON(x)
#endif  // _MSC_VER

// =============================================================================

#endif /* CORE_DETAILS_MACROS_HPP */
