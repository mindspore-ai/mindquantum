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
#include <variant>

#include <symengine/basic.h>
#include <symengine/expression.h>
#include <symengine/real_double.h>
#include <symengine/symengine_exception.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/symengine.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/ops/parametric/angle_base.hpp"
#include "experimental/ops/parametric/config.hpp"
#include "experimental/ops/parametric/register_gate_type.hpp"

#include <catch2/catch_test_macros.hpp>

// =============================================================================

using namespace std::literals::string_view_literals;

namespace {
using mindquantum::ops::parametric::AngleParametricBase;

class OneDoubleGate {
 public:
    static constexpr auto kind() {
        return "test.num.one"sv;
    }

    static constexpr auto num_targets = 1UL;

    explicit OneDoubleGate(double alpha) : alpha_(alpha) {
    }

    MQ_NODISCARD const auto& param() const {
        return alpha_;
    }

 private:
    double alpha_;
};

class AngleGate {
 public:
    static constexpr auto kind() {
        return "test.angle.one"sv;
    }

    explicit AngleGate(double alpha) : alpha_(alpha) {
    }

    MQ_NODISCARD const auto& angle() const {
        return alpha_;
    }

 private:
    double alpha_;
};

class AngleParam : public AngleParametricBase<AngleParam, AngleGate, 4> {
 public:
    static constexpr auto kind() {
        return "test.param.one"sv;
    }

    using base_t::base_t;
};
}  // namespace

namespace parametric = mindquantum::ops::parametric;

// =============================================================================

TEST_CASE("Register non symbolic gate", "[parametric][ops]") {
    auto param = parametric::get_param(OneDoubleGate(3.1415));

    // Make sure that the gate is not known
    CHECK(std::holds_alternative<std::monostate>(param));

    parametric::register_gate_type<OneDoubleGate>();

    param = parametric::get_param(OneDoubleGate(3.1415));
    CHECK(std::holds_alternative<double>(param));

    // --------------------------------

    param = parametric::get_param(AngleGate(3.1415));
    CHECK(std::holds_alternative<std::monostate>(param));

    parametric::register_gate_type<AngleGate>();

    param = parametric::get_param(AngleGate(3.1415));
    CHECK(std::holds_alternative<double>(param));
}

// =============================================================================

TEST_CASE("Register symbolic gate", "[parametric][ops]") {
    using namespace SymEngine;
    auto param = parametric::get_param(AngleParam(symbol("x")));

    // Make sure that the gate is not known
    CHECK(std::holds_alternative<std::monostate>(param));

    parametric::register_gate_type<AngleParam>();

    param = parametric::get_param(AngleParam(symbol("x")));
    CHECK(std::holds_alternative<parametric::param_list_t>(param));
}

// =============================================================================
