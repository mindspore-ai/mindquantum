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

# lint_cmake: -whitespace/indent

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)
  message(FATAL_ERROR "Clang < 7.0.0 is currently not supported!")
endif()
include(CheckCXXSourceCompiles)

# ==============================================================================

# Dummy function to create a new variable scope
function(__test_cxx20_memory)
  set(CMAKE_CXX_STANDARD 20)

  # NB: This below fails with Clang < 9.0.0
  check_cxx_source_compiles(
    [[
#include <memory>
int main() {
  return 0;
}
]]
    compiler_cxx20_memory_works)

  set(_MQ_MEMORY_CXX20_WORKS FALSE)
  if(compiler_cxx20_memory_works)
    set(_MQ_MEMORY_CXX20_WORKS TRUE)
  endif()

  set(_MQ_MEMORY_CXX20_WORKS
      ${_MQ_MEMORY_CXX20_WORKS}
      PARENT_SCOPE)

  set(_MQ_MEMORY_CXX20_WORKS
      ${_MQ_MEMORY_CXX20_WORKS}
      CACHE INTERNAL compiler_cxx20_memory_works)
endfunction()

__test_cxx20_memory()

if(NOT _MQ_MEMORY_CXX20_WORKS)
  set(CMAKE_CXX_STANDARD 17)
endif()

# --------------------------------------

# ~~~
# Check whether some C++ code compiles
#
# check_cxx_code_compiles(<cmake_identifier> <out-var> <cxx_standard> <code>)
# ~~~
function(check_cxx_code_compiles cmake_identifier var cxx_standard code)
  if(cxx_standard MATCHES "cxx_std_([0-9]+)")
    set(CMAKE_CXX_STANDARD ${CMAKE_MATCH_1})
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 20 AND NOT _MQ_MEMORY_CXX20_WORKS)
    set(CMAKE_CXX_STANDARD 17)
  endif()

  if(MSVC)
    get_property(_msvc_flags GLOBAL PROPERTY _compile_msvc_flags_CXX)
    set(CMAKE_REQUIRED_FLAGS ${_msvc_flags})
  endif()

  check_cxx_source_compiles("${code}" "${cmake_identifier}")

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
endfunction()

# ==============================================================================

check_cxx_code_compiles(
  compiler_has_cxx20_operator_not_equal_synthesis
  MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
  cxx_std_20
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

check_cxx_code_compiles(
  compiler_has_implicit_template_deduction_guides
  MQ_HAS_IMPLICIT_TEMPLATE_DEDUCTION_GUIDES
  cxx_std_20
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

check_cxx_code_compiles(
  compiler_has_remove_cvref_t
  MQ_HAS_REMOVE_CVREF_T
  cxx_std_20
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

check_cxx_code_compiles(
  compiler_has_map_contains
  MQ_HAS_MAP_CONTAINS
  cxx_std_20
  [[
#include <map>
int main() { std::map<int, double> m{{0, 1.}, {1, 2.}}; return m.contains(1); }
]])

# --------------------------------------

check_cxx_code_compiles(
  compiler_has_detected_ts2
  MQ_HAS_DETECTED_TS2
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

check_cxx_code_compiles(
  compiler_has_std_filesystem
  MQ_HAS_STD_FILESYSTEM
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

check_cxx_code_compiles(
  compiler_has_class_non_type_template_args
  MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
  cxx_std_20
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

check_cxx_code_compiles(
  compiler_has_concepts
  MQ_HAS_CONCEPTS
  cxx_std_20
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

# --------------------------------------

check_cxx_code_compiles(
  compiler_has_concepts_library
  MQ_HAS_CONCEPT_LIBRARY
  cxx_std_20
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

if(compiler_has_concepts AND NOT compiler_has_concepts_library)
  check_cxx_code_compiles(
    compiler_has_concept_destructible
    MQ_HAS_CONCEPT_DESTRUCTIBLE
    cxx_std_20
    [[
#include <concepts>

template <std::destructible T>
void foo(T* t) { delete t; }
class A {};
int main() {
  auto* a = new A;
  foo(a);
}
]])
elseif(compiler_has_concepts) # C++20 concepts + concepts library
  set(MQ_HAS_CONCEPT_DESTRUCTIBLE TRUE)
else()
  set(MQ_HAS_CONCEPT_DESTRUCTIBLE FALSE)
endif()

# --------------------------------------

check_cxx_code_compiles(
  compiler_has_std_launder
  MQ_HAS_STD_LAUNDER
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

check_cxx_code_compiles(
  compiler_has_constexpr_std_vector
  MQ_HAS_CONSTEXPR_STD_VECTOR
  cxx_std_20
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

# ==============================================================================

configure_file(${CMAKE_CURRENT_LIST_DIR}/cxx20_config.hpp.in ${PROJECT_BINARY_DIR}/core/cxx20_config.hpp)

add_library(cxx20_compat INTERFACE)
target_include_directories(cxx20_compat INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)

# ------------------------------------------------------------------------------

append_to_property(mq_install_targets GLOBAL cxx20_compat)
install(FILES ${PROJECT_BINARY_DIR}/core/cxx20_config.hpp DESTINATION ${MQ_INSTALL_INCLUDEDIR}/experimental/core)

# ==============================================================================
