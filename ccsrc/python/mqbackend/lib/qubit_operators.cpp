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

#include <cstdint>

#include <fmt/format.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "details/define_terms_ops.hpp"
#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/terms_operator_base.hpp"
#include "ops/transform/jordan_wigner.hpp"
#include "ops/transform/parity.hpp"

#include "python/core/boost_multi_index.hpp"

namespace ops = mindquantum::ops;
namespace py = pybind11;
using MT = mindquantum::MT;

void init_qubit_operators(py::module& module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;
    namespace op = bindops::details;

    using pr_t = mq::ParameterResolver<MT>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<MT>>;
    using all_scalar_types_t = std::tuple<MT, std::complex<MT>, pr_t, pr_cmplx_t>;

    // Register empty base class (for instance(X, QubitOperatorBase) purposes
    py::class_<ops::QubitOperatorBase, std::shared_ptr<ops::QubitOperatorBase>>(
        module, "QubitOperatorBase",
        "Base class for all C++ qubit operators. Use only for isinstance(obj, QubitOperatorBase) or use "
        "is_qubit_operator(obj)");
    module.def("is_qubit_operator", &pybind11::isinstance<ops::QubitOperatorBase>);

    // NB: pybind11 maps both float and MT to Python float
    auto [qop_double, qop_cmplx_double, qop_pr_double, qop_pr_cmplx_double]
        = bindops::define_qubit_ops<MT, std::complex<MT>, pr_t, pr_cmplx_t>::apply(
            module, "QubitOperatorD", "QubitOperatorCD", "QubitOperatorPRD", "QubitOperatorPRCD");

    // ---------------------------------

    using QubitOperatorD = decltype(qop_double)::type;
    using QubitOperatorCD = decltype(qop_cmplx_double)::type;
    using QubitOperatorPRD = decltype(qop_pr_double)::type;
    using QubitOperatorPRCD = decltype(qop_pr_cmplx_double)::type;

    using all_qop_types_t = std::tuple<MT, std::complex<MT>, pr_t, pr_cmplx_t, QubitOperatorD, QubitOperatorCD,
                                       QubitOperatorPRD, QubitOperatorPRCD>;

    qop_double.def("cast",
                   bindops::cast<QubitOperatorD, MT, std::complex<MT>, pr_t, pr_cmplx_t, QubitOperatorD,
                                 QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<MT>, ParameterResolver<complex>, "
                   "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorCD, std::complex<MT>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    qop_pr_double.def("cast",
                      bindops::cast<QubitOperatorPRD, MT, std::complex<MT>, pr_t, pr_cmplx_t, QubitOperatorD,
                                    QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<MT>, ParameterResolver<complex>, "
                      "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_pr_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorPRCD, std::complex<MT>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    // ---------------------------------

    qop_double.def_static("simplify", QubitOperatorD::simplify);
    qop_cmplx_double.def_static("simplify", QubitOperatorCD::simplify);
    qop_pr_double.def_static("simplify", QubitOperatorPRD::simplify);
    qop_pr_cmplx_double.def_static("simplify", QubitOperatorPRCD::simplify);

    // ---------------------------------

    using qop_t = decltype(qop_double);
    bindops::binop_definition<op::plus, qop_t>::inplace<MT>(qop_double);
    bindops::binop_definition<op::plus, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::plus, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::inplace<MT>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::times, qop_t>::inplace<MT>(qop_double);
    bindops::binop_definition<op::times, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::times, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::inplace<MT>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::external<all_scalar_types_t>(qop_double);

    using qop_cmplx_t = decltype(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::inplace<MT, std::complex<MT>>(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::inplace<MT, std::complex<MT>>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::inplace<MT, std::complex<MT>>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::inplace<MT, std::complex<MT>>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::external<all_scalar_types_t>(qop_cmplx_double);

    using qop_pr_t = decltype(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::inplace<MT, pr_t>(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::inplace<MT, pr_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::inplace<MT, pr_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::inplace<MT, pr_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::external<all_scalar_types_t>(qop_pr_double);

    using qop_pr_cmplx_t = decltype(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::external<all_scalar_types_t>(qop_pr_cmplx_double);
}
