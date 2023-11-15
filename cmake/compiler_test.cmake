# ==============================================================================
#
# Copyright 2021 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent,-whitespace/extra

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)
  message(FATAL_ERROR "Clang < 7.0.0 is currently not supported!")
endif()
include(CheckCXXSourceCompiles)

is_language_enabled(CUDA _cuda_enabled)
is_language_enabled(NVCXX _nvcxx_enabled)

if(_cuda_enabled)
  include(CheckCUDASourceCompiles)
endif()

# ==============================================================================

# Dummy function to create a new variable scope
function(__test_lang20_memory LANG) # cmake-lint: disable=C0103
  string(TOUPPER "${LANG}" LANG)
  string(TOLOWER "${LANG}" lang)
  set(CMAKE_${LANG}_STANDARD 20)

  # NB: This below fails with Clang < 9.0.0 for C++
  cmake_language(
    CALL
    check_${lang}_source_compiles
    [[
#include <memory>
int main() {
  return 0;
}
]]
    compiler_${lang}20_memory_works)

  set(_MQ_MEMORY_${LANG}20_WORKS FALSE)
  if(compiler_${lang}20_memory_works)
    set(_MQ_MEMORY_${LANG}20_WORKS TRUE)
  endif()

  set(_MQ_MEMORY_${LANG}20_WORKS
      ${_MQ_MEMORY_${LANG}20_WORKS}
      PARENT_SCOPE)

  set(_MQ_MEMORY_${LANG}20_WORKS
      ${_MQ_MEMORY_${LANG}20_WORKS}
      CACHE INTERNAL compiler_${lang}20_memory_works)
endfunction()

__test_lang20_memory(CXX)

if(NOT _MQ_MEMORY_CXX20_WORKS)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(_cuda_enabled)
  __test_lang20_memory(CUDA)
  if(NOT _MQ_MEMORY_CUDA20_WORKS)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

# --------------------------------------

# ~~~
# Check whether some C++ code compiles
#
# check_cxx_code_compiles(<lang> <cmake_identifier> <out-var> <lang_standard> <code> [<lang>, ...])
# ~~~
function(check_code_compiles cmake_identifier var lang_standard code)
  if(NOT "${ARGN}" STREQUAL "")
    set(_lang_list "${ARGN}")
  else()
    set(_lang_list CXX)
    if(_cuda_enabled)
      list(APPEND _lang_list CUDA)
    endif()
  endif()

  set(_cmake_identifier ${cmake_identifier})
  set(_var ${var})

  foreach(_lang ${_lang_list})
    string(TOUPPER "${_lang}" LANG)
    string(TOLOWER "${_lang}" lang)
    set(cmake_identifier "${lang}_${_cmake_identifier}")
    set(var "MQ_${LANG}_${_var}")

    if(lang_standard MATCHES "std_([0-9]+)")
      set(CMAKE_${LANG}_STANDARD ${CMAKE_MATCH_1})
    endif()

    if(CMAKE_${LANG}_STANDARD EQUAL 20 AND NOT _MQ_MEMORY_${LANG}20_WORKS)
      set(CMAKE_${LANG}_STANDARD 17)
    endif()

    if(MSVC)
      get_property(_msvc_flags GLOBAL PROPERTY _compile_msvc_flags_${LANG})
      set(CMAKE_REQUIRED_FLAGS ${_msvc_flags})
    endif()

    cmake_language(CALL check_${lang}_source_compiles "${code}" "${cmake_identifier}")

    set(${var}
        ${${cmake_identifier}}
        PARENT_SCOPE)

    set(${var} FALSE)
    if(${cmake_identifier})
      set(${var} TRUE)
    endif()
    set(${var}
        ${${var}}
        CACHE INTERNAL "${cmake_identifier}")
  endforeach()
endfunction()

# ==============================================================================

check_code_compiles(
  compiler_has_cxx20_operator_not_equal_synthesis
  HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
  std_17
  [[
struct A {
    bool operator==(const A& other) const {
        return false;
    }
};

int main() {
    A a, b;
    return a != b;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_implicit_template_deduction_guides
  HAS_IMPLICIT_TEMPLATE_DEDUCTION_GUIDES
  std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
#include <tuple>

struct A{ int a; };
struct B{ int b; };

template<class... Ts> struct Agg: Ts... {};

int main() {
     Agg agg{A{1}, B{1}};
     return agg.a + agg.b;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_remove_cvref_t
  HAS_REMOVE_CVREF_T
  std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
#include <type_traits>

int main() {
#if __cpp_lib_remove_cvref >= 201711L
   return 0;
#else
   std::remove_cvref_t<const int&> i(0);
   return i;
#endif
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_map_contains
  HAS_MAP_CONTAINS
  std_17
  [[
#include <map>
int main() { std::map<int, double> m{{0, 1.}, {1, 2.}}; return m.contains(1); }
]])

# --------------------------------------

check_code_compiles(
  compiler_has_map_erase_if
  HAS_MAP_ERASE_IF
  std_17
  [[
#include <map>
int main() {
     std::map<int, double> m{{0, 1.}, {1, 2.}};
     return std::erase_if(m, [](const auto& item) {
        auto const& [key, value] = item;
        return (key & 1) == 1;
    });
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_detected_ts2
  HAS_DETECTED_TS2
  cxx_std_17
  [[
#include <experimental/type_traits>
#include <type_traits>
#include <utility>
struct A { int foo() const { return 0; } };
template <typename T> using has_foo = decltype(std::declval<T&>().foo());
static_assert(std::experimental::is_detected_v<has_foo, A>);
int main() { return 0; }
]])

# --------------------------------------

check_code_compiles(
  compiler_has_std_filesystem
  HAS_STD_FILESYSTEM
  cxx_std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
int main() {
#if __cpp_lib_filesystem >= 201703
    return 0;
#else
#error std::filesystem not supported
#endif
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_class_non_type_template_args
  HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
  std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
int main() {
#if __cpp_nontype_template_args >= 201911L
    return 0;
#else
#error C++20 class types and floating-point types in non-type template parameters
#endif
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_concepts
  HAS_CONCEPTS
  std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
int main() {
#if __cpp_concepts >= 201907L
    return 0;
#else
#error C++20 concepts not supported
#endif
}
]])

# if(NOT cxx_compiler_has_concepts) message(WARNING "You are using an older compiler that does not support C++20
# concepts. The code will probably " "compile and run fine, but you should really be upgrading your compiler.") endif()

# --------------------------------------

check_code_compiles(
  compiler_has_concepts_library
  HAS_CONCEPT_LIBRARY
  std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
int main() {
#if __cpp_lib_concepts >= 202002L
    return 0;
#else
#error C++20 standard library concepts not supported
#endif
}
]])

# --------------------------------------

if(cxx_compiler_has_concepts AND NOT cxx_compiler_has_concepts_library)
  check_code_compiles(
    cxx_compiler_has_concept_destructible
    HAS_CONCEPT_DESTRUCTIBLE
    std_17
    [[
#include <concepts>

template <std::destructible T>
void foo(T* t) { delete t; }
class A {};
int main() {
  auto* a = new A;
  foo(a);
}
]]
    CXX)
elseif(cxx_compiler_has_concepts) # C++20 concepts + concepts library
  set(MQ_CXX_HAS_CONCEPT_DESTRUCTIBLE TRUE)
else()
  set(MQ_CXX_HAS_CONCEPT_DESTRUCTIBLE FALSE)
endif()

# --------------------------------------

if(cxx_compiler_has_concepts)
  check_code_compiles(
    compiler_supports_external_dependent_concepts
    SUPPORTS_EXT_DEPENDENT_CONCEPTS
    std_17
    [[
#include <type_traits>

template <typename T, typename U>
concept number = requires(T, U) {
    requires std::is_floating_point_v<T>;
    requires std::is_floating_point_v<U>;
};

template <typename T>
struct A {
    using type = T;

    template <number<type> other_t>
    static int foo();
};

template <typename T>
template <number<typename A<T>::type> other_t>
int A<T>::foo() {
    return 42;
}

int main() {
    return A<double>::foo<float>();
}
]])
else()
  set(MQ_CXX_SUPPORTS_EXT_DEPENDENT_CONCEPTS FALSE)
  set(MQ_CUDA_SUPPORTS_EXT_DEPENDENT_CONCEPTS FALSE)
endif()

# --------------------------------------

check_code_compiles(
  compiler_has_std_launder
  HAS_STD_LAUNDER
  cxx_std_17
  [[
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
#include <new>

int main() {
#if __cpp_lib_launder >= 201606L
    int x[10];
    auto p = std::launder(reinterpret_cast<int(*)[10]>(&x[0]));
#else
#error C++17 standard library launder not supported
#endif
    return 0;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_constexpr_std_vector
  HAS_CONSTEXPR_STD_VECTOR
  std_17
  [[
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif
#include <vector>

#if __cpp_lib_constexpr_vector >= 201907L
constexpr auto foo() { std::vector<int> vec = {1,2,3,4,5}; return vec.size(); }
#endif

int main() {
#if __cpp_lib_constexpr_vector >= 201907L
    return foo();
#else
#    error C++20 constexpr std::vector not supported
#endif
    return 0;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_cxx20_format
  HAS_CXX20_FORMAT
  std_17
  [[
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif
#include <format>
#include <complex>

int main() {
#if __cpp_lib_format >= 201907L
    const auto res = std::format("{}", std::complex<double>{1, 2});
    return res.size();
#else
#    error C++20 std::format not supported
#endif
    return 0;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_cxx20_span
  HAS_CXX20_SPAN
  std_17
  [[
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif
#include <span>
#include <vector>

int main() {
#if __cpp_lib_span >= 202002L
    std::vector<int> vec = {1, 2, 3, 4, 5};
    return std::span<int, 4>{vec}.size();
#else
#    error C++20 std::span not supported
#endif
    return 0;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_has_cxx20_ranges
  HAS_CXX20_RANGES
  std_17
  [[
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

int main() {
#if __cpp_lib_ranges >= 201911L
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::partial_sum(v.cbegin(), v.cend(), v.begin());
    std::ranges::copy(v, std::ostream_iterator<int>(std::cout, " "));
    auto divisible_by = [](int d) { return [d](int m) { return m % d == 0; }; };
    if (std::ranges::any_of(v, divisible_by(7))) {
        std::cout << "At least one number is divisible by 7" << std::endl;
    }
#else
#    error C++20 ranges not supported
#endif
    return 0;
}
]])

# --------------------------------------

check_code_compiles(
  compiler_std_accumulate_use_move
  STD_ACCUMULATE_USE_MOVE
  std_17
  [[
#include <numeric>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return std::accumulate(begin(v), end(v), int{0},
                           [](int&& init, int value) -> decltype(auto) { return init += value; });
    return 0;
}
]])

# ==============================================================================

# NB: second condition is workardoung for Clang < 9.0
if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES AND CMAKE_CXX_STANDARD EQUAL 20)
  target_compile_features(CXX_mindquantum INTERFACE cxx_std_20)
else()
  target_compile_features(CXX_mindquantum INTERFACE cxx_std_17)
endif()
set_target_properties(CXX_mindquantum PROPERTIES CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)
  if(cuda_std_20 IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    target_compile_features(CUDA_mindquantum INTERFACE cuda_std_20)
  elseif(cuda_std_17 IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    target_compile_features(CUDA_mindquantum INTERFACE cuda_std_17)
  else()
    target_compile_features(CUDA_mindquantum INTERFACE cuda_std_14)
  endif()
  set_target_properties(CUDA_mindquantum PROPERTIES CUDA_STANDARD_REQUIRED ON)
endif()

# ------------------------------------------------------------------------------

set(_lang_list CXX)
if(_cuda_enabled)
  list(APPEND _lang_list CUDA)
endif()

set(_configured_headers)
foreach(_lang ${_lang_list})
  set(LANG ${_lang})
  string(TOLOWER "${_lang}" lang)
  set(MQ_DEFINE_MACRO "#cmakedefine01")
  configure_file(${CMAKE_CURRENT_LIST_DIR}/lang20_config.h.in.in ${PROJECT_BINARY_DIR}/${lang}20_config.h.in @ONLY)
  configure_file(${PROJECT_BINARY_DIR}/${lang}20_config.h.in ${PROJECT_BINARY_DIR}/config/${lang}20_config.h)
  list(APPEND _configured_headers "${PROJECT_BINARY_DIR}/config/${lang}20_config.h")

  add_library(${lang}20_compat INTERFACE)
  target_include_directories(${lang}20_compat INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
  append_to_property(mq_install_targets GLOBAL ${lang}20_compat)
endforeach()

# ------------------------------------------------------------------------------

install(FILES ${_configured_headers} DESTINATION ${MQ_INSTALL_INCLUDEDIR}/config)

# ==============================================================================
