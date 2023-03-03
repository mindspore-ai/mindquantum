/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "python/device/binding.hpp"

#include <complex>
#include <memory>

#include <fmt/format.h>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config/constexpr_type_name.hpp"
#include "config/format/parameter_resolver.hpp"
#include "config/format/std_complex.hpp"
#include "config/type_traits.hpp"

#include "core/mq_base_types.hpp"
#include "core/parameter_resolver.hpp"
#include "core/sparse/algo.hpp"
#include "core/sparse/csrhdmatrix.hpp"
#include "core/sparse/paulimat.hpp"
#include "core/two_dim_matrix.hpp"
#include "details/define_terms_ops.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"
#include "ops/gates.hpp"
#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/term_value.hpp"
#include "ops/gates/terms_operator_base.hpp"
#include "ops/hamiltonian.hpp"
#include "ops/transform/bravyi_kitaev.hpp"
#include "ops/transform/bravyi_kitaev_superfast.hpp"
#include "ops/transform/jordan_wigner.hpp"
#include "ops/transform/parity.hpp"
#include "ops/transform/ternary_tree.hpp"

#include "python/core/boost_multi_index.hpp"
#include "python/core/sparse/csrhdmatrix.hpp"
#include "python/details/create_from_container_class.hpp"
#include "python/details/define_binary_operator_helpers.hpp"
#include "python/ops/basic_gate.hpp"
#include "python/ops/build_env.hpp"

namespace py = pybind11;

using mindquantum::sparse::Csr_Plus_Csr;
using mindquantum::sparse::GetPauliMat;
using mindquantum::sparse::PauliMat;
using mindquantum::sparse::PauliMatToCsrHdMatrix;
using mindquantum::sparse::SparseHamiltonian;
using mindquantum::sparse::TransposeCsrHdMatrix;
namespace ops = mindquantum::ops;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

template <typename T>
void init_fermion_operators(py::module &module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;

    using pr_t = mq::ParameterResolver<T>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<T>>;
    using all_scalar_types_t = std::tuple<T, std::complex<T>, pr_t, pr_cmplx_t>;

    // NB: pybind11 maps both float and T to Python float
    auto [fop_double, fop_cmplx_double, fop_pr_double, fop_pr_cmplx_double]
        = bindops::define_fermion_ops<T, std::complex<T>, pr_t, pr_cmplx_t>::apply(
            module, "FermionOperatorD", "FermionOperatorCD", "FermionOperatorPRD", "FermionOperatorPRCD");

    // ---------------------------------

    using FermionOperatorD = typename decltype(fop_double)::type;
    using FermionOperatorCD = typename decltype(fop_cmplx_double)::type;
    using FermionOperatorPRD = typename decltype(fop_pr_double)::type;
    using FermionOperatorPRCD = typename decltype(fop_pr_cmplx_double)::type;

    using all_fop_types_t = std::tuple<T, std::complex<T>, pr_t, pr_cmplx_t, FermionOperatorD, FermionOperatorCD,
                                       FermionOperatorPRD, FermionOperatorPRCD>;

    fop_double.def("cast",
                   bindops::cast<FermionOperatorD, T, std::complex<T>, pr_t, pr_cmplx_t, FermionOperatorD,
                                 FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<T>, ParameterResolver<complex>, "
                   "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_cmplx_double.def(
        "cast", bindops::cast<FermionOperatorCD, std::complex<T>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    fop_pr_double.def("cast",
                      bindops::cast<FermionOperatorPRD, T, std::complex<T>, pr_t, pr_cmplx_t, FermionOperatorD,
                                    FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<T>, ParameterResolver<complex>, "
                      "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_pr_cmplx_double.def(
        "cast", bindops::cast<FermionOperatorPRCD, std::complex<T>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    // ---------------------------------
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRD, int, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRD, T, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRD, const pr_t &, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, int, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, T, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, std::complex<T>, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, const pr_t &, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, const pr_cmplx_t &, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRD &, const FermionOperatorPRCD &, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRD &, int, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRD &, T, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRD &, std::complex<T>, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRD &, const pr_t &, +);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRD &, const pr_cmplx_t &, +);

    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, int, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, T, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, std::complex<T>, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, const pr_t &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, const pr_cmplx_t &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, FermionOperatorPRCD, const FermionOperatorPRD &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, int, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, T, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, std::complex<T>, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, const pr_t &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, const pr_cmplx_t &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const FermionOperatorPRCD &, const FermionOperatorPRD &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRCD &, int, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRCD &, T, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRCD &, std::complex<T>, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRCD &, const pr_t &, +);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const FermionOperatorPRCD &, const pr_cmplx_t &, +);

    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRD, int, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRD, T, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRD, const pr_t &, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, int, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, T, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, std::complex<T>, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, const pr_t &, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, const pr_cmplx_t &, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRD &, const FermionOperatorPRCD &, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRD &, int, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRD &, T, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRD &, std::complex<T>, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRD &, const pr_t &, -);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRD &, const pr_cmplx_t &, -);

    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, int, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, T, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, std::complex<T>, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, const pr_t &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, const pr_cmplx_t &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, FermionOperatorPRCD, const FermionOperatorPRD &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, int, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, T, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, std::complex<T>, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, const pr_t &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, const pr_cmplx_t &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const FermionOperatorPRCD &, const FermionOperatorPRD &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRCD &, int, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRCD &, T, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRCD &, std::complex<T>, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRCD &, const pr_t &, -);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const FermionOperatorPRCD &, const pr_cmplx_t &, -);

    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRD, int, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRD, T, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRD, const pr_t &, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, int, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, T, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, std::complex<T>, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, const pr_t &, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, const pr_cmplx_t &, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRD &, const FermionOperatorPRCD &, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRD &, int, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRD &, T, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRD &, std::complex<T>, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRD &, const pr_t &, *);
    fop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRD &, const pr_cmplx_t &, *);

    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, int, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, T, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, std::complex<T>, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, const pr_t &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, const pr_cmplx_t &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, FermionOperatorPRCD, const FermionOperatorPRD &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, int, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, T, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, std::complex<T>, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, const pr_t &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, const pr_cmplx_t &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const FermionOperatorPRCD &, const FermionOperatorPRD &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRCD &, int, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRCD &, T, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRCD &, std::complex<T>, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRCD &, const pr_t &, *);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const FermionOperatorPRCD &, const pr_cmplx_t &, *);

    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRD, int, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRD, T, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRD, const pr_t &, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRD &, int, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRD &, T, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRD &, std::complex<T>, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRD &, const pr_t &, /);
    fop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRD &, const pr_cmplx_t &, /);

    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRCD, int, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRCD, T, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRCD, std::complex<T>, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRCD, const pr_t &, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, FermionOperatorPRCD, const pr_cmplx_t &, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRCD &, int, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRCD &, T, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRCD &, std::complex<T>, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRCD &, const pr_t &, /);
    fop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const FermionOperatorPRCD &, const pr_cmplx_t &, /);
}

template <typename T>
void init_qubit_operators(py::module &module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;

    using pr_t = mq::ParameterResolver<T>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<T>>;
    using all_scalar_types_t = std::tuple<T, std::complex<T>, pr_t, pr_cmplx_t>;

    // NB: pybind11 maps both float and T to Python float
    auto [qop_double, qop_cmplx_double, qop_pr_double, qop_pr_cmplx_double]
        = bindops::define_qubit_ops<T, std::complex<T>, pr_t, pr_cmplx_t>::apply(
            module, "QubitOperatorD", "QubitOperatorCD", "QubitOperatorPRD", "QubitOperatorPRCD");

    // ---------------------------------

    using QubitOperatorD = typename decltype(qop_double)::type;
    using QubitOperatorCD = typename decltype(qop_cmplx_double)::type;
    using QubitOperatorPRD = typename decltype(qop_pr_double)::type;
    using QubitOperatorPRCD = typename decltype(qop_pr_cmplx_double)::type;

    using all_qop_types_t = std::tuple<T, std::complex<T>, pr_t, pr_cmplx_t, QubitOperatorD, QubitOperatorCD,
                                       QubitOperatorPRD, QubitOperatorPRCD>;

    qop_double.def("cast",
                   bindops::cast<QubitOperatorD, T, std::complex<T>, pr_t, pr_cmplx_t, QubitOperatorD, QubitOperatorCD,
                                 QubitOperatorPRD, QubitOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<T>, ParameterResolver<complex>, "
                   "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorCD, std::complex<T>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    qop_pr_double.def("cast",
                      bindops::cast<QubitOperatorPRD, T, std::complex<T>, pr_t, pr_cmplx_t, QubitOperatorD,
                                    QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<T>, ParameterResolver<complex>, "
                      "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_pr_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorPRCD, std::complex<T>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    // ---------------------------------

    qop_double.def_static("simplify", QubitOperatorD::simplify);
    qop_cmplx_double.def_static("simplify", QubitOperatorCD::simplify);
    qop_pr_double.def_static("simplify", QubitOperatorPRD::simplify);
    qop_pr_cmplx_double.def_static("simplify", QubitOperatorPRCD::simplify);

    // ---------------------------------
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRD, int, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRD, T, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRD, const pr_t &, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, int, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, T, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, std::complex<T>, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, const pr_t &, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, const pr_cmplx_t &, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRD &, const QubitOperatorPRCD &, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRD &, int, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRD &, T, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRD &, std::complex<T>, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRD &, const pr_t &, +);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRD &, const pr_cmplx_t &, +);

    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, int, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, T, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, std::complex<T>, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, const pr_t &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, const pr_cmplx_t &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(add, QubitOperatorPRCD, const QubitOperatorPRD &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, int, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, T, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, std::complex<T>, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, const pr_t &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, const pr_cmplx_t &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(add, const QubitOperatorPRCD &, const QubitOperatorPRD &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRCD &, int, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRCD &, T, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRCD &, std::complex<T>, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRCD &, const pr_t &, +);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(add, const QubitOperatorPRCD &, const pr_cmplx_t &, +);

    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRD, int, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRD, T, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRD, const pr_t &, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, int, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, T, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, std::complex<T>, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, const pr_t &, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, const pr_cmplx_t &, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRD &, const QubitOperatorPRCD &, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRD &, int, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRD &, T, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRD &, std::complex<T>, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRD &, const pr_t &, -);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRD &, const pr_cmplx_t &, -);

    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, int, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, T, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, std::complex<T>, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, const pr_t &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, const pr_cmplx_t &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(sub, QubitOperatorPRCD, const QubitOperatorPRD &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, int, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, T, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, std::complex<T>, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, const pr_t &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, const pr_cmplx_t &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(sub, const QubitOperatorPRCD &, const QubitOperatorPRD &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRCD &, int, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRCD &, T, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRCD &, std::complex<T>, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRCD &, const pr_t &, -);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(sub, const QubitOperatorPRCD &, const pr_cmplx_t &, -);

    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRD, int, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRD, T, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRD, const pr_t &, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, int, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, T, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, std::complex<T>, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, const pr_t &, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, const pr_cmplx_t &, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRD &, const QubitOperatorPRCD &, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRD &, int, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRD &, T, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRD &, std::complex<T>, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRD &, const pr_t &, *);
    qop_pr_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRD &, const pr_cmplx_t &, *);

    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, int, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, T, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, std::complex<T>, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, const pr_t &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, const pr_cmplx_t &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(mul, QubitOperatorPRCD, const QubitOperatorPRD &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, int, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, T, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, std::complex<T>, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, const pr_t &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, const pr_cmplx_t &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(mul, const QubitOperatorPRCD &, const QubitOperatorPRD &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRCD &, T, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRCD &, std::complex<T>, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRCD &, const pr_t &, *);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_REV(mul, const QubitOperatorPRCD &, const pr_cmplx_t &, *);

    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRD, int, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRD, T, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRD, const pr_t &, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRD &, int, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRD &, T, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRD &, std::complex<T>, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRD &, const pr_t &, /);
    qop_pr_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRD &, const pr_cmplx_t &, /);

    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRCD, int, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRCD, T, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRCD, std::complex<T>, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRCD, const pr_t &, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_INPLACE(truediv, QubitOperatorPRCD, const pr_cmplx_t &, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRCD &, int, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRCD &, T, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRCD &, std::complex<T>, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRCD &, const pr_t &, /);
    qop_pr_cmplx_double.PYBIND11_DEFINE_BINOP_EXT(truediv, const QubitOperatorPRCD &, const pr_cmplx_t &, /);
}

template <typename T>
void init_transform(py::module &module) {  // NOLINT(runtime/references)
    using namespace pybind11::literals;    // NOLINT(build/namespaces_literals)

    namespace transform = mindquantum::ops::transform;

    using bindops::fop_t;
    using bindops::qop_t;
    using pr_t = mindquantum::ParameterResolver<T>;
    using pr_cmplx_t = mindquantum::ParameterResolver<std::complex<T>>;

    module.def("parity", &transform::parity<fop_t<T>>, "ops"_a, "n_qubits"_a);
    module.def("parity", &transform::parity<fop_t<std::complex<T>>>, "ops"_a, "n_qubits"_a);
    module.def("parity", &transform::parity<fop_t<pr_t>>, "ops"_a, "n_qubits"_a);
    module.def("parity", &transform::parity<fop_t<pr_cmplx_t>>, "ops"_a, "n_qubits"_a);

    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<T>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<std::complex<T>>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<pr_t>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<pr_cmplx_t>>);

    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<T>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<std::complex<T>>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<pr_t>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<pr_cmplx_t>>);

    module.def("bravyi_kitaev", &transform::bravyi_kitaev<fop_t<T>>, "ops"_a, "n_qubits"_a);
    module.def("bravyi_kitaev", &transform::bravyi_kitaev<fop_t<std::complex<T>>>, "ops"_a, "n_qubits"_a);
    module.def("bravyi_kitaev", &transform::bravyi_kitaev<fop_t<pr_t>>, "ops"_a, "n_qubits"_a);
    module.def("bravyi_kitaev", &transform::bravyi_kitaev<fop_t<pr_cmplx_t>>, "ops"_a, "n_qubits"_a);

    module.def("bravyi_kitaev_superfast", &transform::bravyi_kitaev_superfast<fop_t<T>>);
    module.def("bravyi_kitaev_superfast", &transform::bravyi_kitaev_superfast<fop_t<std::complex<T>>>);
    module.def("bravyi_kitaev_superfast", &transform::bravyi_kitaev_superfast<fop_t<pr_t>>);
    module.def("bravyi_kitaev_superfast", &transform::bravyi_kitaev_superfast<fop_t<pr_cmplx_t>>);

    module.def("ternary_tree", &transform::ternary_tree<fop_t<T>>, "ops"_a, "n_qubits"_a);
    module.def("ternary_tree", &transform::ternary_tree<fop_t<std::complex<T>>>, "ops"_a, "n_qubits"_a);
    module.def("ternary_tree", &transform::ternary_tree<fop_t<pr_t>>, "ops"_a, "n_qubits"_a);
    module.def("ternary_tree", &transform::ternary_tree<fop_t<pr_cmplx_t>>, "ops"_a, "n_qubits"_a);
}

template <typename T>
void init_terms_operators(pybind11::module &module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;

    using pr_t = mq::ParameterResolver<T>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<T>>;

    /* These types are used when one wants to replace some parameters inside a FermionOperator or QubitOperator.
     * The two types for T and std::complex<T> do not do anything in practice but are defined anyway in order
     * to have a consistent API.
     */
    py::class_<ops::details::CoeffSubsProxy<T>>(module, "DoubleSubsProxy",
                                                "Substitution proxy class for floating point numbers")
        .def(py::init<T>());
    py::class_<ops::details::CoeffSubsProxy<std::complex<T>>>(module, "CmplxDoubleSubsProxy",
                                                              "Substitution proxy class for complex numbers")
        .def(py::init<std::complex<T>>());
    py::class_<ops::details::CoeffSubsProxy<pr_t>>(module, "DoublePRSubsProxy",
                                                   "Substitution proxy class for mqbackend.real_pr")
        .def(py::init<pr_t>());
    py::class_<ops::details::CoeffSubsProxy<pr_cmplx_t>>(module, "CmplxPRSubsProxy",
                                                         "Substitution proxy class for mqbackend.complex_pr")
        .def(py::init<pr_cmplx_t>());

    // -----------------------------------------------------------------------------

    init_fermion_operators<T>(module);
    init_qubit_operators<T>(module);

    py::module trans = module.def_submodule("transform", "MindQuantum-C++ operators transform");
    init_transform<T>(trans);
}

template <typename T>
auto BindPR(py::module &module, const std::string &name) {  // NOLINT(runtime/references)
    using mindquantum::MST;
    using mindquantum::ParameterResolver;
    using mindquantum::SS;
    using mindquantum::python::create_from_python_container_class_with_trampoline;
    using pr_t = mindquantum::ParameterResolver<T>;

    // NB: this below is required because of GCC < 9
    using factory_func_t = decltype(&create_from_python_container_class_with_trampoline<pr_t, MST<T>>);
    using cast_complex_func_t = decltype(&pr_t::template Cast<mindquantum::traits::to_cmplx_type_t<T>>);

    using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

    auto klass
        = py::class_<pr_t, std::shared_ptr<pr_t>>(module, name.c_str())
              // ------------------------------
              // Constructors
              .def(py::init<T>())
              .def(py::init<std::string>())
              .def(py::init<int>())
              .def(py::init<const MST<T> &>(), "data"_a)
              .def(py::init<const MST<T> &, T>(), "data"_a, "coeff"_a)
              .def(py::init<const MST<T> &, T, const SS &, const SS &>())
              // ------------------------------
              // Properties
              .def_property_readonly("const", [](const pr_t &pr) { return pr.const_value; })
              .def_readonly("data", &pr_t::data_)
              .def_property_readonly(
                  "is_complex", [](const pr_t &) constexpr { return mindquantum::traits::is_complex_v<T>; })
              .def_property_readonly("encoder_parameters", [](const pr_t &pr) { return pr.encoder_parameters_; })
              .def_property_readonly("no_grad_parameters", [](const pr_t &pr) { return pr.no_grad_parameters_; })
              // ------------------------------
              // Member functions
              .def("ansatz_part", &pr_t::AnsatzPart)
              .def("as_ansatz", &pr_t::AsAnsatz)
              .def("as_encoder", &pr_t::AsEncoder)
              .def("combination", &pr_t::Combination)
              .def("cast_complex",
                   static_cast<cast_complex_func_t>(&pr_t::template Cast<mindquantum::traits::to_cmplx_type_t<T>>))
              .def("conjugate", &pr_t::Conjugate)
              .def("display", &pr_t::PrintInfo)
              .def("encoder_part", &pr_t::EncoderPart)
              .def("get_key", &pr_t::GetKey)
              .def("imag", &pr_t::Imag)
              .def("is_anti_hermitian", &pr_t::IsAntiHermitian)
              .def("is_const", &pr_t::IsConst)
              .def("is_hermitian", &pr_t::IsHermitian)
              .def("keep_imag", &pr_t::KeepImag)
              .def("keep_real", &pr_t::KeepReal)
              .def("no_grad", &pr_t::NoGrad)
              .def("no_grad_part", &pr_t::NoGradPart)
              .def("params_name", &pr_t::ParamsName)
              .def("pop", &pr_t::Pop)
              .def("real", &pr_t::Real)
              .def("requires_grad", &pr_t::RequiresGrad)
              .def("requires_grad_part", &pr_t::RequiresGradPart)
              .def("set_const", &pr_t::SetConst)
              .def("size", &pr_t::Size)
              .def("update", &pr_t::template Update<T>)
              // ------------------------------
              // Python magic methods
              .def("__bool__", &pr_t::IsNotZero)
              .def("__contains__", &pr_t::Contains)
              .def("__copy__", &pr_t::Copy)
              .def("__getitem__", py::overload_cast<const std::string &>(&pr_t::GetItem, py::const_))
              .def("__getitem__", py::overload_cast<size_t>(&pr_t::GetItem, py::const_))
              .def("__len__", &pr_t::Size)
              .def("__repr__",
                   [](const pr_t &pr) {
                       return fmt::format("ParameterResolver<{}>({})", mindquantum::get_type_name<T>(), pr.ToString());
                   })
              .def("__setitem__", &pr_t::SetItem)
              .def("__str__", &pr_t::ToString)
              // ------------------------------
              // Python arithmetic operators
              .def(py::self == T())
              .def(py::self == py::self)
              .def(-py::self);

    if constexpr (mindquantum::traits::is_std_complex_v<T>) {
        using real_t = mindquantum::traits::to_real_type_t<T>;
        klass.def(py::init<MST<real_t>>())
            .def(py::init<const mindquantum::MST<real_t> &>(), "data"_a)
            .def(py::init<const mindquantum::MST<real_t> &, real_t>(), "data"_a, "coeff"_a);
    }

    // NB: VERY* important: these overload below needs to be the LAST
    // NB2: the cast is only required for older compilers (typically GCC < 9)
    klass.def(
        pybind11::init(static_cast<factory_func_t>(&create_from_python_container_class_with_trampoline<pr_t, MST<T>>)),
        "py_class"_a, "Constructor from the encapsulating Python class (using a _cpp_obj attribute)");

    pybind11::implicitly_convertible<pybind11::object, pr_t>();
    return klass;
}

namespace mindquantum::python {
void init_logging(pybind11::module &module);  // NOLINT(runtime/references)NOLINT
}  // namespace mindquantum::python

void BindTypeIndependentGate(py::module &module) {  // NOLINT(runtime/references)
    using mindquantum::Index;
    using mindquantum::VT;
    py::class_<mindquantum::MeasureGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::MeasureGate>>(
        module, "MeasureGate")
        .def(py::init<std::string, const VT<Index> &>(), "name"_a, "obj_qubits"_a);
    py::class_<mindquantum::IGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::IGate>>(module, "IGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::XGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::XGate>>(module, "XGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::YGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::YGate>>(module, "YGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::ZGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::ZGate>>(module, "ZGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::HGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::HGate>>(module, "HGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::ISWAPGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::ISWAPGate>>(module,
                                                                                                        "ISWAPGate")
        .def(py::init<bool, const VT<Index> &, const VT<Index> &>(), "daggered"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SWAPGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SWAPGate>>(module,
                                                                                                      "SWAPGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SGate>>(module, "SGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SdagGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SdagGate>>(module,
                                                                                                      "SdagGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::TGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::TGate>>(module, "TGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::TdagGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::TdagGate>>(module,
                                                                                                      "TdagGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PauliChannel, mindquantum::BasicGate, std::shared_ptr<mindquantum::PauliChannel>>(
        module, "PauliChannel")
        .def(py::init<double, double, double, const VT<Index> &, const VT<Index> &>(), "px"_a, "py"_a, "pz"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::AmplitudeDampingChannel, mindquantum::BasicGate,
               std::shared_ptr<mindquantum::AmplitudeDampingChannel>>(module, "AmplitudeDampingChannel")
        .def(py::init<double, const VT<Index> &, const VT<Index> &>(), "damping_coeff"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PhaseDampingChannel, mindquantum::BasicGate,
               std::shared_ptr<mindquantum::PhaseDampingChannel>>(module, "PhaseDampingChannel")
        .def(py::init<double, const VT<Index> &, const VT<Index> &>(), "damping_coeff"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
}
template <typename T>
void BindTypeDependentGate(py::module &module) {  // NOLINT(runtime/references)
    using mindquantum::CT;
    using mindquantum::Index;
    using mindquantum::ParameterResolver;
    using mindquantum::VT;
    using mindquantum::VVT;
    py::class_<mindquantum::RXGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RXGate<T>>>(module,
                                                                                                        "RXGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RYGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RYGate<T>>>(module,
                                                                                                        "RYGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RZGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RZGate<T>>>(module,
                                                                                                        "RZGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxxGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxxGate<T>>>(module,
                                                                                                          "RxxGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RyyGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RyyGate<T>>>(module,
                                                                                                          "RyyGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RzzGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RzzGate<T>>>(module,
                                                                                                          "RzzGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxyGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxyGate<T>>>(module,
                                                                                                          "RxyGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxzGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxzGate<T>>>(module,
                                                                                                          "RxzGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RyzGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::RyzGate<T>>>(module,
                                                                                                          "RyzGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::GPGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::GPGate<T>>>(module,
                                                                                                        "GPGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PSGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::PSGate<T>>>(module,
                                                                                                        "PSGate")
        .def(py::init<const ParameterResolver<T> &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::U3<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::U3<T>>>(module, "u3")
        .def(py::init<const ParameterResolver<T> &, const ParameterResolver<T> &, const ParameterResolver<T> &,
                      const VT<Index> &, const VT<Index> &>(),
             "theta"_a, "phi"_a, "lambda"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::FSim<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::FSim<T>>>(module, "fsim")
        .def(py::init<const ParameterResolver<T> &, const ParameterResolver<T> &, const VT<Index> &,
                      const VT<Index> &>(),
             "theta"_a, "phi"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::KrausChannel<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::KrausChannel<T>>>(
        module, "KrausChannel")
        .def(py::init<const VT<VVT<CT<T>>> &, const VT<Index> &, const VT<Index> &>(), "kraus_operator_set"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::CustomGate<T>, mindquantum::BasicGate, std::shared_ptr<mindquantum::CustomGate<T>>>(
        module, "CustomGate")
        .def(py::init<std::string, uint64_t, uint64_t, int, const ParameterResolver<T>, const VT<Index> &,
                      const VT<Index> &>(),
             "name"_a, "m_addr"_a, "dm_addr"_a, "dim"_a, "pr"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>())
        .def(py::init<std::string, const mindquantum::Dim2Matrix<T> &, const VT<Index> &, const VT<Index> &>(),
             "name"_a, "mat"_a, "obj_qubits"_a, "ctrl_qubits"_a);
}
template <typename T>
auto BindOther(py::module &module) {
    using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
    using mindquantum::CT;
    using mindquantum::Dim2Matrix;
    using mindquantum::Hamiltonian;
    using mindquantum::Index;
    using mindquantum::ParameterResolver;
    using mindquantum::PauliTerm;
    using mindquantum::VS;
    using mindquantum::VT;
    using mindquantum::VVT;
    using mindquantum::python::CsrHdMatrix;
    // matrix
    py::class_<Dim2Matrix<T>, std::shared_ptr<Dim2Matrix<T>>>(module, "dim2matrix")
        .def(py::init<>())
        .def(py::init<const VVT<CT<T>> &>())
        .def("PrintInfo", &Dim2Matrix<T>::PrintInfo);
    py::module float_gate = module.def_submodule("gate", "MindQuantum-C++ gate");
    BindTypeDependentGate<T>(float_gate);
    // parameter resolver

    auto real_pr = BindPR<T>(module, "real_pr");
    auto complex_pr = BindPR<std::complex<T>>(module, "complex_pr");

    namespace op = bindops::details;

    using real_pr_t = decltype(real_pr);
    using pr_t = typename real_pr_t::type;
    using complex_pr_t = decltype(complex_pr);
    using pr_cmplx_t = typename complex_pr_t::type;

    using all_scalar_types_t = std::tuple<T, std::complex<T>, pr_t, pr_cmplx_t>;

    complex_pr.def("update", &pr_cmplx_t::template Update<T>);

    bindops::binop_definition<op::plus, real_pr_t>::template inplace<T, pr_t>(real_pr);
    bindops::binop_definition<op::plus, real_pr_t>::template external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::plus, real_pr_t>::template reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::template inplace<T, pr_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::template external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::template reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::template inplace<T, pr_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::template external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::template reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::template inplace<T, pr_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::template external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::template reverse<all_scalar_types_t>(real_pr);

    bindops::binop_definition<op::plus, complex_pr_t>::template inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::plus, complex_pr_t>::template external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::plus, complex_pr_t>::template reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::template inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::template external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::template reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::template inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::template external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::template reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::template inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::template external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::template reverse<all_scalar_types_t>(complex_pr);

    // pauli mat
    py::class_<PauliMat<T>, std::shared_ptr<PauliMat<T>>>(module, "pauli_mat")
        .def(py::init<>())
        .def(py::init<const PauliTerm<T> &, Index>())
        .def_readonly("n_qubits", &PauliMat<T>::n_qubits_)
        .def_readonly("dim", &PauliMat<T>::dim_)
        .def_readwrite("coeff", &PauliMat<T>::p_)
        .def("PrintInfo", &PauliMat<T>::PrintInfo);

    module.def("get_pauli_mat", &GetPauliMat<T>);

    // csr_hd_matrix
    py::class_<CsrHdMatrix<T>, std::shared_ptr<CsrHdMatrix<T>>>(module, "csr_hd_matrix")
        .def(py::init<>())
        .def(py::init<Index, Index, py::array_t<Index>, py::array_t<Index>, py::array_t<CT<T>>>())
        .def("PrintInfo", &CsrHdMatrix<T>::PrintInfo);
    module.def("csr_plus_csr", &Csr_Plus_Csr<T>);
    module.def("transpose_csr_hd_matrix", &TransposeCsrHdMatrix<T>);
    module.def("pauli_mat_to_csr_hd_matrix", &PauliMatToCsrHdMatrix<T>);

    // hamiltonian
    py::class_<Hamiltonian<T>, std::shared_ptr<Hamiltonian<T>>>(module, "hamiltonian")
        .def(py::init<>())
        .def(py::init<const VT<PauliTerm<T>> &>())
        .def(py::init<const VT<PauliTerm<T>> &, Index>())
        .def(py::init<std::shared_ptr<CsrHdMatrix<T>>, Index>())
        .def_readwrite("how_to", &Hamiltonian<T>::how_to_)
        .def_readwrite("n_qubits", &Hamiltonian<T>::n_qubits_)
        .def_readwrite("ham", &Hamiltonian<T>::ham_)
        .def_readwrite("ham_sparse_main", &Hamiltonian<T>::ham_sparse_main_)
        .def_readwrite("ham_sparse_second", &Hamiltonian<T>::ham_sparse_second_);
    module.def("sparse_hamiltonian", &SparseHamiltonian<T>);

    // NB: needs to be *after* declaration of ParameterResolver to pybind11
    init_terms_operators<T>(module);
}

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
    m.doc() = "MindQuantum C++ plugin";

    py::module logging = m.def_submodule("logging", "MindQuantum-C++ logging module");
    mindquantum::python::init_logging(logging);

    auto gate_id = py::enum_<mindquantum::GateID>(m, "GateID")
                       .value("I", mindquantum::GateID::I)
                       .value("X", mindquantum::GateID::X)
                       .value("Y", mindquantum::GateID::Y)
                       .value("Z", mindquantum::GateID::Z)
                       .value("RX", mindquantum::GateID::RX)
                       .value("RY", mindquantum::GateID::RY)
                       .value("RZ", mindquantum::GateID::RZ)
                       .value("Rxx", mindquantum::GateID::Rxx)
                       .value("Ryy", mindquantum::GateID::Ryy)
                       .value("Rzz", mindquantum::GateID::Rzz)
                       .value("H", mindquantum::GateID::H)
                       .value("SWAP", mindquantum::GateID::SWAP)
                       .value("ISWAP", mindquantum::GateID::ISWAP)
                       .value("T", mindquantum::GateID::T)
                       .value("Tdag", mindquantum::GateID::T)
                       .value("S", mindquantum::GateID::Sdag)
                       .value("Sdag", mindquantum::GateID::Sdag)
                       .value("CNOT", mindquantum::GateID::CNOT)
                       .value("CZ", mindquantum::GateID::CZ)
                       .value("GP", mindquantum::GateID::GP)
                       .value("PS", mindquantum::GateID::PS)
                       .value("U3", mindquantum::GateID::U3)
                       .value("FSim", mindquantum::GateID::FSim)
                       .value("M", mindquantum::GateID::M)
                       .value("PL", mindquantum::GateID::PL)
                       .value("AD", mindquantum::GateID::AD)
                       .value("PD", mindquantum::GateID::PD)
                       .value("KRAUS", mindquantum::GateID::KRAUS)
                       .value("CUSTOM", mindquantum::GateID::CUSTOM);
    gate_id.attr("__repr__") = pybind11::cpp_function(
        [](const mindquantum::GateID &id) -> pybind11::str { return fmt::format("GateID.{}", id); },
        pybind11::name("name"), pybind11::is_method(gate_id));
    gate_id.attr("__str__") = pybind11::cpp_function(
        [](const mindquantum::GateID &id) -> pybind11::str { return fmt::format("{}", id); }, pybind11::name("name"),
        pybind11::is_method(gate_id));
    auto term_value = py::enum_<mindquantum::ops::TermValue>(m, "TermValue")
                          .value("I", mindquantum::ops::TermValue::I)
                          .value("X", mindquantum::ops::TermValue::X)
                          .value("Y", mindquantum::ops::TermValue::Y)
                          .value("Z", mindquantum::ops::TermValue::Z)
                          .value("a", mindquantum::ops::TermValue::a)
                          .value("adg", mindquantum::ops::TermValue::adg)
                          .def(
                              "__lt__",
                              [](const mindquantum::ops::TermValue &lhs, const mindquantum::ops::TermValue &rhs)
                                  -> bool { return static_cast<uint8_t>(lhs) < static_cast<uint8_t>(rhs); },
                              pybind11::is_operator());

    term_value.attr("__repr__") = pybind11::cpp_function(
        [](const mindquantum::ops::TermValue &value) -> pybind11::str { return fmt::format("TermValue.{}", value); },
        pybind11::name("name"), pybind11::is_method(term_value));
    term_value.attr("__str__") = pybind11::cpp_function(
        [](const mindquantum::ops::TermValue &value) -> pybind11::str { return fmt::format("{}", value); },
        pybind11::name("name"), pybind11::is_method(term_value));

    // Register empty base class (for instance(X, FermionOperatorBase) purposes
    py::class_<ops::FermionOperatorBase, std::shared_ptr<ops::FermionOperatorBase>>(
        m, "FermionOperatorBase",
        "Base class for all C++ fermion operators. Use only for isinstance(obj, FermionOperatorBase) or use "
        "is_fermion_operator(obj)");

    // Register empty base class (for instance(X, QubitOperatorBase) purposes
    py::class_<ops::QubitOperatorBase, std::shared_ptr<ops::QubitOperatorBase>>(
        m, "QubitOperatorBase",
        "Base class for all C++ qubit operators. Use only for isinstance(obj, QubitOperatorBase) or use "
        "is_qubit_operator(obj)");

    m.attr("EQ_TOLERANCE") = py::float_(ops::details::EQ_TOLERANCE);

    m.def("is_qubit_operator", &pybind11::isinstance<ops::QubitOperatorBase>);

    m.def("is_fermion_operator", &pybind11::isinstance<ops::FermionOperatorBase>);

    py::module mqbackend_double = m.def_submodule("double", "MindQuantum-C++ double backend");
    py::module mqbackend_float = m.def_submodule("float", "MindQuantum-C++ float backend");
    py::module gate = m.def_submodule("gate", "MindQuantum-C++ gate");
    py::class_<mindquantum::BasicGate, std::shared_ptr<mindquantum::BasicGate>>(gate, "BasicGate").def(py::init<>());
    BindTypeIndependentGate(gate);
    // py::module double_gate = mqbackend_double.def_submodule("gate", "MindQuantum-C++ gate");
    // py::module float_gate = mqbackend_float.def_submodule("gate", "MindQuantum-C++ gate");
    // BindTypeDependentGate<double>(double_gate);
    // BindTypeDependentGate<float>(float_gate);

    BindOther<double>(mqbackend_double);
    BindOther<float>(mqbackend_float);

    py::module c = m.def_submodule("c", "pybind11 c++ env");
    mindquantum::BindPybind11Env(c);

    py::module device = m.def_submodule("device", "Quantum device module");
    BindTopology(device);
}
