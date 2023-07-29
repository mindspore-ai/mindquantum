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

#include <map>
#include <optional>

#include "config/common_type.h"
#include "math/pr/parameter_resolver.h"

#include <catch2/catch_test_macros.h>

// =============================================================================

namespace mq = mindquantum::traits;

template <typename float_t>
using cmplx_t = std::complex<float_t>;

template <typename float_t>
using pr_t = mindquantum::ParameterResolver<float_t>;

// -----------------------------------------------------------------------------
// Cases identical to std::common_type

static_assert(std::is_same_v<std::common_type_t<float, float>, mq::common_type_t<float, float>>);
static_assert(std::is_same_v<std::common_type_t<float, double>, mq::common_type_t<float, double>>);
static_assert(std::is_same_v<std::common_type_t<double, double>, mq::common_type_t<double, double>>);
static_assert(std::is_same_v<std::common_type_t<cmplx_t<float>, cmplx_t<float>>,
                             mq::common_type_t<cmplx_t<float>, cmplx_t<float>>>);
static_assert(std::is_same_v<std::common_type_t<cmplx_t<double>, cmplx_t<double>>,
                             mq::common_type_t<cmplx_t<double>, cmplx_t<double>>>);
static_assert(std::is_same_v<std::common_type_t<float, cmplx_t<float>>, mq::common_type_t<float, cmplx_t<float>>>);
static_assert(std::is_same_v<std::common_type_t<cmplx_t<float>, cmplx_t<double>>,
                             mq::common_type_t<cmplx_t<float>, cmplx_t<double>>>);
static_assert(std::is_same_v<std::common_type_t<double, cmplx_t<double>>, mq::common_type_t<double, cmplx_t<double>>>);

static_assert(std::is_same_v<std::common_type_t<float, double, cmplx_t<double>>,
                             mq::common_type_t<float, double, cmplx_t<double>>>);
static_assert(
    std::is_same_v<std::common_type_t<float, double, int, char>, mq::common_type_t<float, double, int, char>>);

// -----------------------------------------------------------------------------
// Cases different from std::common_type

static_assert(std::is_same_v<cmplx_t<double>, mq::common_type_t<double, cmplx_t<float>>>);
static_assert(std::is_same_v<cmplx_t<double>, mq::common_type_t<float, cmplx_t<double>>>);
static_assert(std::is_same_v<cmplx_t<double>, mq::common_type_t<cmplx_t<float>, int, float, double, cmplx_t<float>>>);

// =============================================================================

static_assert(std::is_same_v<pr_t<double>, mq::common_type_t<double, pr_t<float>>>);
static_assert(std::is_same_v<pr_t<double>, mq::common_type_t<float, pr_t<double>>>);
static_assert(std::is_same_v<pr_t<double>, mq::common_type_t<pr_t<float>, int, float, double, pr_t<float>>>);
static_assert(std::is_same_v<pr_t<cmplx_t<float>>, mq::common_type_t<float, cmplx_t<float>, pr_t<float>>>);
static_assert(std::is_same_v<pr_t<cmplx_t<double>>, mq::common_type_t<cmplx_t<double>, pr_t<float>>>);

// =============================================================================

TEST_CASE("Empty") {
}
