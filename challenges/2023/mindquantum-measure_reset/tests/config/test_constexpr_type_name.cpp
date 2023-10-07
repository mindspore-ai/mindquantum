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

#include "config/constexpr_type_name.h"

#include <catch2/catch_test_macros.h>

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
using mindquantum::get_type_name;

// -----------------------------------------------------------------------------

namespace sub = mindquantum::sub;

static_assert(get_type_name<void>() == "void"sv);
static_assert(get_type_name<bool>() == "bool"sv);
static_assert(get_type_name<char>() == "char"sv);
static_assert(get_type_name<int>() == "int"sv);
static_assert(get_type_name<mindquantum::A>() == "mindquantum::A"sv);
static_assert(get_type_name<mindquantum::B>() == "mindquantum::B"sv);
static_assert(get_type_name<mindquantum::C>() == "mindquantum::C"sv);
static_assert(get_type_name<sub::D<double>>() == "mindquantum::sub::D<double>"sv);

#if (defined __clang__) && (MQ_CLANG_MAJOR > 10)
static_assert(get_type_name<sub::E<sub::D<int>>>() == "mindquantum::sub::E<mindquantum::sub::D<int>>"sv);
#elif (defined _MSC_VER) && !(defined __clang__)  // NOLINT(whitespace/parens)
static_assert(get_type_name<sub::E<sub::D<int>>>() == "mindquantum::sub::E<class mindquantum::sub::D<int> >"sv);
#else
static_assert(get_type_name<sub::E<sub::D<int>>>() == "mindquantum::sub::E<mindquantum::sub::D<int> >"sv);
#endif

// =============================================================================

TEST_CASE("Empty") {
}
