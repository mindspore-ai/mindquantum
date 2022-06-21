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

#ifndef MACROS_HPP
#define MACROS_HPP

#ifdef __clang__
// clang-format off
#     define CLANG_DIAG_DO_PRAGMA(x) _Pragma(#x)
// clang-format on
#    define CLANG_DIAG_PRAGMA(x) CLANG_DIAG_DO_PRAGMA(GCC diagnostic x)
#    define CLANG_DIAG_OFF(x)                                                                                          \
        CLANG_DIAG_PRAGMA(push)                                                                                        \
        CLANG_DIAG_PRAGMA(ignored x)
#    define CLANG_DIAG_ON(x) CLANG_DIAG_PRAGMA(pop)
#else
#    define CLANG_DIAG_OFF(x)
#    define CLANG_DIAG_ON(x)
#endif  // __clang__

#if (defined __GNUC__) && !(defined __clang__)
// clang-format off
#     define GCC_DIAG_DO_PRAGMA(x) _Pragma(#x)
// clang-format on
#    define GCC_DIAG_PRAGMA(x) GCC_DIAG_DO_PRAGMA(GCC diagnostic x)
#    define GCC_DIAG_OFF(x)                                                                                            \
        GCC_DIAG_PRAGMA(push)                                                                                          \
        GCC_DIAG_PRAGMA(ignored x)
#    define GCC_DIAG_ON(x) GCC_DIAG_PRAGMA(pop)
#else
#    define GCC_DIAG_OFF(x)
#    define GCC_DIAG_ON(x)
#endif  // __GNUC__

#ifdef _MSC_VER
#    define MSVC_DIAG_OFF(x) __pragma(warning(push)) __pragma(warning(disable : warningNumber))
#    define MSVC_DIAG_ON(x)  __pragma(warning(pop))
#else
#    define MSVC_DIAG_OFF(x)
#    define MSVC_DIAG_ON(x)
#endif  // _MSC_VER

#endif /* MACROS_HPP */
