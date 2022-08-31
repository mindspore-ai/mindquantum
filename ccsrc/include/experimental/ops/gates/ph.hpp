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

#ifndef PH_HPP
#define PH_HPP

#include <string_view>

#include <tweedledum/IR/Operator.h>
#include <tweedledum/Utils/Matrix.h>

namespace mindquantum::ops {
class Ph {
    using UMatrix2 = tweedledum::UMatrix2;

 public:
    static constexpr std::string_view kind() {
        return "projectq.ph";
    }

    static constexpr auto num_params = 1UL;

    explicit Ph(double angle) : angle_(angle) {
    }

    Ph adjoint() const {
        return Ph(-angle_);
    }

    constexpr bool is_symmetric() const {
        return true;
    }

    UMatrix2 const matrix() const {
        using Complex = tweedledum::Complex;
        Complex const a = std::exp(Complex(0., angle_));
        return (UMatrix2() << a, 0., 0., a).finished();
    }

    bool operator==(Ph const& other) const {
        return angle_ == other.angle_;
    }

    const auto& angle() const {
        return angle_;
    }

 private:
    double const angle_;
};
}  // namespace mindquantum::ops

#endif /* PH_HPP */
