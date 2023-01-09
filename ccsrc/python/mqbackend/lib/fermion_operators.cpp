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

void init_fermion_operators(py::module& module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;
    namespace op = bindops::details;

    using pr_t = mq::ParameterResolver<MT>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<MT>>;
    using all_scalar_types_t = std::tuple<MT, std::complex<MT>, pr_t, pr_cmplx_t>;

    // Register empty base class (for instance(X, FermionOperatorBase) purposes
    py::class_<ops::FermionOperatorBase, std::shared_ptr<ops::FermionOperatorBase>>(
        module, "FermionOperatorBase",
        "Base class for all C++ fermion operators. Use only for isinstance(obj, FermionOperatorBase) or use "
        "is_fermion_operator(obj)");
    module.def("is_fermion_operator", &pybind11::isinstance<ops::FermionOperatorBase>);

    // NB: pybind11 maps both float and MT to Python float
    auto [fop_double, fop_cmplx_double, fop_pr_double, fop_pr_cmplx_double]
        = bindops::define_fermion_ops<MT, std::complex<MT>, pr_t, pr_cmplx_t>::apply(
            module, "FermionOperatorD", "FermionOperatorCD", "FermionOperatorPRD", "FermionOperatorPRCD");

    // ---------------------------------

    using FermionOperatorD = decltype(fop_double)::type;
    using FermionOperatorCD = decltype(fop_cmplx_double)::type;
    using FermionOperatorPRD = decltype(fop_pr_double)::type;
    using FermionOperatorPRCD = decltype(fop_pr_cmplx_double)::type;

    using all_fop_types_t = std::tuple<MT, std::complex<MT>, pr_t, pr_cmplx_t, FermionOperatorD, FermionOperatorCD,
                                       FermionOperatorPRD, FermionOperatorPRCD>;

    fop_double.def("cast",
                   bindops::cast<FermionOperatorD, MT, std::complex<MT>, pr_t, pr_cmplx_t, FermionOperatorD,
                                 FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<MT>, ParameterResolver<complex>, "
                   "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_cmplx_double.def(
        "cast", bindops::cast<FermionOperatorCD, std::complex<MT>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    fop_pr_double.def("cast",
                      bindops::cast<FermionOperatorPRD, MT, std::complex<MT>, pr_t, pr_cmplx_t, FermionOperatorD,
                                    FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<MT>, ParameterResolver<complex>, "
                      "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_pr_cmplx_double.def(
        "cast",
        bindops::cast<FermionOperatorPRCD, std::complex<MT>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    // ---------------------------------

    using fop_t = decltype(fop_double);
    bindops::binop_definition<op::plus, fop_t>::inplace<MT>(fop_double);
    bindops::binop_definition<op::plus, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::plus, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::inplace<MT>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::times, fop_t>::inplace<MT>(fop_double);
    bindops::binop_definition<op::times, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::times, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::inplace<MT>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::external<all_scalar_types_t>(fop_double);

    using fop_cmplx_t = decltype(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::inplace<MT, std::complex<MT>>(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::inplace<MT, std::complex<MT>>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::inplace<MT, std::complex<MT>>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::inplace<MT, std::complex<MT>>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);

    using fop_pr_t = decltype(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::inplace<MT, pr_t>(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::inplace<MT, pr_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::inplace<MT, pr_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::inplace<MT, pr_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);

    using fop_pr_cmplx_t = decltype(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
}
