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

#ifndef PARAM_NAMES_HPP
#define PARAM_NAMES_HPP

#include <complex>
#include <string_view>

#include <symengine/eval_double.h>

#include "experimental/ops/parametric/config.hpp"

namespace mindquantum::ops::parametric {
namespace details {
//! Defines a real parameter
struct real_tag_t {  // NOLINT(altera-struct-pack-align)
    using type = double;
    static auto eval(const basic_t& expr) {
        return eval_double(*expr);
    }
};
//! Defines a complex parameter
struct complex_tag_t {  // NOLINT(altera-struct-pack-align)
    using type = std::complex<double>;
    static auto eval(const basic_t& expr) {
        return eval_complex_double(*expr);
    }
};
}  // namespace details

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DEFINE_PARAM_STRUCT(type, param_name)                                                                          \
    struct param_name { /* NOLINT(altera-struct-pack-align) */                                                         \
        static constexpr std::string_view name = #param_name;                                                          \
        using param_type = type;                                                                                       \
    }

namespace real {
DEFINE_PARAM_STRUCT(details::real_tag_t, alpha);
DEFINE_PARAM_STRUCT(details::real_tag_t, beta);
DEFINE_PARAM_STRUCT(details::real_tag_t, gamma);
DEFINE_PARAM_STRUCT(details::real_tag_t, delta);
DEFINE_PARAM_STRUCT(details::real_tag_t, epsilon);
DEFINE_PARAM_STRUCT(details::real_tag_t, zeta);
DEFINE_PARAM_STRUCT(details::real_tag_t, eta);
DEFINE_PARAM_STRUCT(details::real_tag_t, theta);
DEFINE_PARAM_STRUCT(details::real_tag_t, iota);
DEFINE_PARAM_STRUCT(details::real_tag_t, kappa);
DEFINE_PARAM_STRUCT(details::real_tag_t, lambda);
DEFINE_PARAM_STRUCT(details::real_tag_t, mu);
DEFINE_PARAM_STRUCT(details::real_tag_t, nu);
DEFINE_PARAM_STRUCT(details::real_tag_t, xi);
DEFINE_PARAM_STRUCT(details::real_tag_t, omicron);
DEFINE_PARAM_STRUCT(details::real_tag_t, pi);
DEFINE_PARAM_STRUCT(details::real_tag_t, rho);
DEFINE_PARAM_STRUCT(details::real_tag_t, sigma);
DEFINE_PARAM_STRUCT(details::real_tag_t, tau);
DEFINE_PARAM_STRUCT(details::real_tag_t, upsilon);
DEFINE_PARAM_STRUCT(details::real_tag_t, phi);
DEFINE_PARAM_STRUCT(details::real_tag_t, chi);
DEFINE_PARAM_STRUCT(details::real_tag_t, omega);
}  // namespace real

namespace complex {
DEFINE_PARAM_STRUCT(details::complex_tag_t, alpha);
DEFINE_PARAM_STRUCT(details::complex_tag_t, beta);
DEFINE_PARAM_STRUCT(details::complex_tag_t, gamma);
DEFINE_PARAM_STRUCT(details::complex_tag_t, delta);
DEFINE_PARAM_STRUCT(details::complex_tag_t, epsilon);
DEFINE_PARAM_STRUCT(details::complex_tag_t, zeta);
DEFINE_PARAM_STRUCT(details::complex_tag_t, eta);
DEFINE_PARAM_STRUCT(details::complex_tag_t, theta);
DEFINE_PARAM_STRUCT(details::complex_tag_t, iota);
DEFINE_PARAM_STRUCT(details::complex_tag_t, kappa);
DEFINE_PARAM_STRUCT(details::complex_tag_t, lambda);
DEFINE_PARAM_STRUCT(details::complex_tag_t, mu);
DEFINE_PARAM_STRUCT(details::complex_tag_t, nu);
DEFINE_PARAM_STRUCT(details::complex_tag_t, xi);
DEFINE_PARAM_STRUCT(details::complex_tag_t, omicron);
DEFINE_PARAM_STRUCT(details::complex_tag_t, pi);
DEFINE_PARAM_STRUCT(details::complex_tag_t, rho);
DEFINE_PARAM_STRUCT(details::complex_tag_t, sigma);
DEFINE_PARAM_STRUCT(details::complex_tag_t, tau);
DEFINE_PARAM_STRUCT(details::complex_tag_t, upsilon);
DEFINE_PARAM_STRUCT(details::complex_tag_t, phi);
DEFINE_PARAM_STRUCT(details::complex_tag_t, chi);
DEFINE_PARAM_STRUCT(details::complex_tag_t, omega);
}  // namespace complex

#undef DEFINE_PARAM_STRUCT
}  // namespace mindquantum::ops::parametric

#endif /* PARAM_NAMES_HPP */
