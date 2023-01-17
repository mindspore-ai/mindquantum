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

#ifndef BOOST_X3_COMPLEX_NUMBER_HPP
#define BOOST_X3_COMPLEX_NUMBER_HPP

#include <complex>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

#include "core/mq_base_types.hpp"

namespace mindquantum::ast {
template <typename T>
struct complex_number {
    T real;
    T imag = 0;

    operator std::complex<T>() const& {
        return {real, imag};
    }
    operator std::complex<T>() && {
        return {real, imag};
    }
};
}  // namespace mindquantum::ast
BOOST_FUSION_ADAPT_STRUCT(mindquantum::ast::complex_number<mindquantum::MT>, real, imag);

namespace mindquantum::parser {
namespace boost_x3 = boost::spirit::x3;
struct imag_unit_class {};
const boost_x3::rule<imag_unit_class, boost_x3::unused_type> imag_unit = "Imaginary unit ('i', 'j', 'I', 'J')";
const auto imag_unit_def = boost_x3::char_("ijIJ");

struct complex_class {};
const boost_x3::rule<complex_class, ast::complex_number<MT>> complex = "Complex number ('X', 'Yj', or '(X+Yj)')";
static const auto complex_def = ('(' >> boost_x3::double_ >> -(boost_x3::double_ >> imag_unit) >> ')')
                                // | (x3::double_ >> x3::double_ >> imag_unit)
                                | (boost_x3::attr(0.) >> boost_x3::double_ >> imag_unit)
                                | (boost_x3::double_ >> boost_x3::attr(0.));

BOOST_SPIRIT_DEFINE(imag_unit, complex);
}  // namespace mindquantum::parser

#endif /* BOOST_X3_COMPLEX_NUMBER_HPP */
