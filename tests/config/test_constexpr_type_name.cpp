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

#include <map>
#include <optional>

#include "config/constexpr_type_name.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// =============================================================================

namespace mindquantum {
class A {};
struct B {};
class C : public A {};

namespace sub {
template <typename T>
class D {};
template <typename U>
struct E {};
}  // namespace sub
}  // namespace mindquantum

// =============================================================================

using namespace std::literals::string_view_literals;
using mindquantum::get_relative_type_name;
using mindquantum::get_type_name;

// -----------------------------------------------------------------------------

static_assert(get_type_name<void>() == "void"sv);
static_assert(get_type_name<bool>() == "bool"sv);
static_assert(get_type_name<char>() == "char"sv);
static_assert(get_type_name<int>() == "int"sv);
static_assert(get_type_name<mindquantum::A>() == "mindquantum::A"sv);
static_assert(get_type_name<mindquantum::B>() == "mindquantum::B"sv);
static_assert(get_type_name<mindquantum::C>() == "mindquantum::C"sv);
static_assert(get_type_name<mindquantum::sub::D<double>>() == "mindquantum::sub::D<double>"sv);
#if (defined __clang__) && (MQ_CLANG_MAJOR >= 10)
static_assert(get_type_name<mindquantum::sub::E<std::optional<int>>>() == "mindquantum::sub::E<std::optional<int>>"sv);
#else
static_assert(get_type_name<mindquantum::sub::E<std::optional<int>>>() == "mindquantum::sub::E<std::optional<int> >"sv);
#endif

// -----------------------------------------------------------------------------

namespace mindquantum {
struct marker_t;
}  // namespace mindquantum

using marker_t = mindquantum::marker_t;

static_assert(get_relative_type_name<void, marker_t>() == get_type_name<void>());
static_assert(get_relative_type_name<bool, marker_t>() == get_type_name<bool>());
static_assert(get_relative_type_name<char, marker_t>() == get_type_name<char>());
static_assert(get_relative_type_name<int, marker_t>() == get_type_name<int>());
static_assert(get_relative_type_name<mindquantum::A, marker_t>() == "A"sv);
static_assert(get_relative_type_name<mindquantum::B, marker_t>() == "B"sv);
static_assert(get_relative_type_name<mindquantum::C, marker_t>() == "C"sv);
static_assert(get_relative_type_name<mindquantum::sub::D<double>, marker_t>() == "sub::D<double>"sv);
#if (defined __clang__) && (MQ_CLANG_MAJOR >= 10)
static_assert(get_relative_type_name<mindquantum::sub::E<std::optional<int>>, marker_t>()
              == "sub::E<std::optional<int>>"sv);
#else
static_assert(get_relative_type_name<mindquantum::sub::E<std::optional<int>>, marker_t>()
              == "sub::E<std::optional<int> >"sv);
#endif

// =============================================================================

TEST_CASE("Empty") {
}
