//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include <string_view>

#include <symengine/basic.h>
#include <symengine/expression.h>
#include <symengine/real_double.h>
#include <symengine/symengine_exception.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/symengine.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/ops/parametric/angle_base.hpp"

#include <catch2/catch_test_macros.hpp>

#if __has_include(<numbers>) && __cplusplus > 201703L
static constexpr auto PI_VAL = std::numbers::pi;
#else
static constexpr auto PI_VAL = 3.141592653589793;
#endif  // __has_include(<numbers>) && C++20

// =============================================================================

using namespace std::literals::string_view_literals;

namespace {
using mindquantum::ops::parametric::AngleParametricBase;

class NumOne {
 public:
    static constexpr auto kind() {
        return "test.num.one"sv;
    }

    static constexpr auto num_targets = 1UL;

    explicit NumOne(double alpha) : alpha_(alpha) {
    }

    MQ_NODISCARD const auto& angle() const {
        return alpha_;
    }

 private:
    double alpha_;
};

class ParamOne : public AngleParametricBase<ParamOne, NumOne, 4> {
 public:
    static constexpr auto kind() {
        return "test.param.one"sv;
    }

    using base_t::base_t;
};
}  // namespace

// =============================================================================

TEST_CASE("AngleParametricGate", "[parametric][ops]") {
    using namespace SymEngine;
    namespace real = mindquantum::ops::parametric::real;

    Expression x{symbol("x")};
    Expression y{symbol("y")};
    double x_num(42);
    double y_num(-314.15);

    SymEngine::map_basic_basic subs;
    subs[x] = SymEngine::number(x_num);
    subs[y] = SymEngine::number(y_num);

    ParamOne gate1{x};
    ParamOne gate1_bis{x};
    ParamOne gate1_ter{neg(-x)};
    ParamOne gate2{x + y};

    static_assert(gate1.num_params == 1);
    static_assert(gate1.param_name(0) == real::theta::name);

    SECTION("Basic") {
        CHECK(eq(*gate1.param(0), x));
        REQUIRE(eq(*gate2.param(0), x + y));

        REQUIRE(gate1 == gate1_ter);

        const auto gate1_num = gate1.eval_full(subs);
        CHECK(gate1_num.kind() == ParamOne::non_param_type::kind());
        CHECK(gate1_num.kind() == NumOne::kind());
        REQUIRE(gate1_num.angle() == std::fmod(x_num, PI_VAL * 4));

        const auto gate2_num = gate2.eval_full(subs);
        CHECK(gate2_num.kind() == ParamOne::non_param_type::kind());
        CHECK(gate2_num.kind() == NumOne::kind());
        REQUIRE(gate2_num.angle() == std::fmod(x_num + y_num, PI_VAL * 4));
    }

    SECTION("Adjoint") {
        ParamOne gate1_inv{-x};
        ParamOne gate2_inv{-x - y};

        CHECK(gate1.adjoint() == gate1_inv);
        REQUIRE(gate2.adjoint() == gate2_inv);
    }
}

// =============================================================================
