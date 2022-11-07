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

#include "experimental/core/operator_traits.hpp"
#include "experimental/ops/parametric/gate_base.hpp"

#include <catch2/catch_test_macros.hpp>

// =============================================================================

using mindquantum::catch2::Equals;

namespace {
using mindquantum::ops::parametric::ParametricBase;
namespace real = mindquantum::ops::parametric::real;

class NumOne {
 public:
    static constexpr auto kind() {
        return "test.num.one";
    }

    static constexpr auto num_targets = 1UL;

    explicit NumOne(double alpha) : alpha_(alpha) {
    }

    MQ_NODISCARD const auto& alpha() const {
        return alpha_;
    }

 private:
    double alpha_;
};

class ParamOne : public ParametricBase<ParamOne, NumOne, real::alpha> {
 public:
    static constexpr auto kind() {
        return "test.param.one";
    }

    template <typename evald_param_t>
    static auto to_param_type(const ParamOne& /* other */, evald_param_t&& evald_param) {
        return ParamOne{std::forward<evald_param_t>(evald_param)};
    }

    template <typename evald_param_t>
    static auto to_non_param_type(const ParamOne& /* other */, evald_param_t&& evald_param) {
        return NumOne{std::forward<evald_param_t>(evald_param)};
    }

    using base_t::base_t;
};

class NumTwo {
 public:
    static constexpr auto kind() {
        return "test.num.two";
    }

    static constexpr auto num_targets = 1UL;

    NumTwo(double alpha, double beta) : alpha_(alpha), beta_(beta) {
    }

    MQ_NODISCARD const auto& alpha() const {
        return alpha_;
    }
    MQ_NODISCARD const auto& beta() const {
        return beta_;
    }

 private:
    double alpha_;
    double beta_;
};

class ParamTwo : public ParametricBase<ParamTwo, NumTwo, real::alpha, real::beta> {
 public:
    static constexpr auto kind() {
        return "test.param.two";
    }

    template <typename... evald_params_t>
    static auto to_param_type(const ParamTwo& /* other */, evald_params_t&&... evald_params) {
        return ParamTwo{std::forward<evald_params_t>(evald_params)...};
    }

    template <typename... evald_params_t>
    static auto to_non_param_type(const ParamTwo& /* other */, evald_params_t&&... evald_params) {
        return NumTwo{std::forward<evald_params_t>(evald_params)...};
    }

    using base_t::base_t;
};
}  // namespace

// =============================================================================

static_assert(::ParamOne::has_const_num_targets);
static_assert(::ParamTwo::has_const_num_targets);

static_assert(mindquantum::traits::num_targets<::NumOne> == 1UL);
static_assert(mindquantum::traits::num_targets<::ParamOne> == 1UL);
static_assert(mindquantum::traits::num_targets<::NumTwo> == 1UL);
static_assert(mindquantum::traits::num_targets<::ParamTwo> == 1UL);

// =============================================================================

TEST_CASE("ParametricGate/Basic", "[parametric][ops]") {
    using namespace SymEngine;
    namespace real = mindquantum::ops::parametric::real;

    Expression x{symbol("x")}, y{symbol("y")}, z{symbol("z")};

    SECTION("Single parameter") {
        ParamOne gate1{x};
        ParamOne gate1_bis{x};
        static_assert(gate1.num_params == 1);
        static_assert(gate1.param_name(0) == real::alpha::name);

        CHECK(eq(*gate1.param(0), x));
        REQUIRE(!eq(*gate1.param(0), y));

        ParamOne gate2{neg(x + y)};
        CHECK(!eq(*gate1.param(0), *gate2.param(0)));
        REQUIRE(eq(*gate2.param(0), expand(-(x + y))));

        CHECK(gate1 == gate1);
        CHECK(gate1 == gate1_bis);
        CHECK(gate1 != gate2);
        CHECK(gate2 != gate1);
        REQUIRE(gate2 == gate2);

        ParamOne gate2_bis{-x - y};
        std::cerr << str(*expand(gate2.param(0))) << ", " << str(*expand(gate2_bis.param(0))) << '\n';
        REQUIRE(gate2 == gate2_bis);
        REQUIRE(gate2_bis == gate2);

        REQUIRE_THAT(gate1.params(), Equals(x));
        REQUIRE_THAT(gate2.params(), Equals(expand(-(x + y))));

        ParamOne num1{12};
        CHECK(is_a<Integer>(*num1.param(0)));
        REQUIRE(eq(*num1.param(0), *SymEngine::integer(12)));
        ParamOne num2{-3.1415};
        CHECK(is_a_Number(*num1.param(0)));
        REQUIRE(eq(*num2.param(0), *SymEngine::number(-3.1415)));
    }

    SECTION("Two parameters") {
        ParamTwo gate1{x, y};
        ParamTwo gate1_bis{x, y};
        static_assert(gate1.num_params == 2);
        static_assert(gate1.param_name(0) == real::alpha::name);
        static_assert(gate1.param_name(1) == real::beta::name);

        const auto param_ref = std::vector<Expression>{x, y};
        CHECK_THAT(gate1.params(), Equals(x, y));

        ParamTwo gate2{x + y, y};

        CHECK(gate1 == gate1);
        CHECK(gate1 == gate1_bis);
        CHECK(gate1 != gate2);
        CHECK(gate2 != gate1);
        REQUIRE(gate2 == gate2);
    }
}

// -----------------------------------------------------------------------------

TEST_CASE("ParametricGate/Evaluation", "[parametric][ops]") {
    using namespace SymEngine;

    Expression x{symbol("x")}, y{symbol("y")}, z{symbol("z")};
    int x_num(42);
    double y_num(-3.1415), z_num(1.23);

    ParamOne gate1{x};
    ParamTwo gate2{x + y, y};
    ParamTwo gate3{x + y, y + z};

    CHECK_THAT(gate1.params(), Equals(x));
    CHECK_THAT(gate2.params(), Equals(expand(x + y), y));
    CHECK_THAT(gate3.params(), Equals(expand(x + y), expand(y + z)));

    SymEngine::map_basic_basic subs;
    subs[x] = SymEngine::integer(x_num);
    subs[y] = SymEngine::number(y_num);

    SymEngine::map_basic_basic subs2;
    subs2[z] = SymEngine::number(z_num);

    const auto gate1_num = gate1.eval_full(subs);
    CHECK(gate1_num.kind() == ParamOne::non_param_type::kind());
    CHECK(gate1_num.kind() == NumOne::kind());
    REQUIRE(gate1_num.alpha() == x_num);

    const auto gate2_num = gate2.eval_full(subs);
    CHECK(gate2_num.kind() == ParamTwo::non_param_type::kind());
    CHECK(gate2_num.kind() == NumTwo::kind());
    REQUIRE(gate2_num.alpha() == x_num + y_num);
    REQUIRE(gate2_num.beta() == y_num);

    REQUIRE_THROWS_AS(gate3.eval_full(subs), SymEngine::SymEngineException);

    const auto gate3_partial = gate3.eval(subs);
    CHECK(gate3_partial.kind() == gate3.kind());
    CHECK_THAT(gate3_partial.params(), Equals(SymEngine::number(x_num + y_num), y_num + z));

    const auto gate3_num = gate3_partial.eval_full(subs2);
    CHECK(gate3_num.kind() == ParamTwo::non_param_type::kind());
    CHECK(gate3_num.kind() == NumTwo::kind());
    REQUIRE(gate3_num.alpha() == x_num + y_num);
    REQUIRE(gate3_num.beta() == y_num + z_num);
}

// =============================================================================
