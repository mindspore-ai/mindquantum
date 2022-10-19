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

#include "config/constexpr_type_name.hpp"
#include "config/format/parameter_resolver.hpp"
#include "config/format/std_complex.hpp"
#include "config/type_traits.hpp"

#include "ops/basic_gate.hpp"

#ifdef ENABLE_PROJECTQ
#    include "projectq.h"
#endif

#include "core/mq_base_types.hpp"
#include "core/parameter_resolver.hpp"
#include "core/sparse/algo.hpp"
#include "core/sparse/csrhdmatrix.hpp"
#include "core/sparse/paulimat.hpp"
#include "core/two_dim_matrix.hpp"
#include "ops/gates.hpp"
#include "ops/hamiltonian.hpp"

#include "python/core/sparse/csrhdmatrix.hpp"
#include "python/details/create_from_container_class.hpp"
#include "python/details/define_binary_operator_helpers.hpp"
#include "python/ops/basic_gate.hpp"

namespace py = pybind11;

using mindquantum::sparse::Csr_Plus_Csr;
using mindquantum::sparse::GetPauliMat;
using mindquantum::sparse::PauliMat;
using mindquantum::sparse::PauliMatToCsrHdMatrix;
using mindquantum::sparse::SparseHamiltonian;
using mindquantum::sparse::TransposeCsrHdMatrix;

template <typename T>
auto BindPR(py::module &module, const std::string &name) {  // NOLINT(runtime/references)
    using mindquantum::MST;
    using mindquantum::ParameterResolver;
    using mindquantum::SS;
    using mindquantum::python::create_from_python_container_class_with_trampoline;
#ifdef ENABLE_PROJECTQ
    using mindquantum::projectq::InnerProduct;
    using mindquantum::projectq::Projectq;
#endif

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
              .def("encoder_parameters", [](const pr_t &pr) { return pr.encoder_parameters_; })
              .def("encoder_part", &pr_t::EncoderPart)
              .def("get_key", &pr_t::GetKey)
              .def("imag", &pr_t::Imag)
              .def("is_anti_hermitian", &pr_t::IsAntiHermitian)
              .def("is_const", &pr_t::IsConst)
              .def("is_hermitian", &pr_t::IsHermitian)
              .def("keep_imag", &pr_t::KeepImag)
              .def("keep_real", &pr_t::KeepReal)
              .def("no_grad", &pr_t::NoGrad)
              .def("no_grad_parameters", [](const pr_t &pr) { return pr.no_grad_parameters_; })
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
void init_logging(pybind11::module &module);  // NOLINT(runtime/references)
}  // namespace mindquantum::python

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
    using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
    using mindquantum::CT;
    using mindquantum::Dim2Matrix;
    using mindquantum::GetGateByName;
    using mindquantum::GetMeasureGate;
    using mindquantum::Hamiltonian;
    using mindquantum::Index;
    using mindquantum::MT;
    using mindquantum::ParameterResolver;
    using mindquantum::PauliTerm;
    using mindquantum::VS;
    using mindquantum::VT;
    using mindquantum::VVT;
    using mindquantum::python::BasicGate;
    using mindquantum::python::CsrHdMatrix;

    m.doc() = "MindQuantum C++ plugin";

    py::module logging = m.def_submodule("logging", "MindQuantum-C++ logging module");
    mindquantum::python::init_logging(logging);

    // matrix
    py::class_<Dim2Matrix<MT>, std::shared_ptr<Dim2Matrix<MT>>>(m, "dim2matrix")
        .def(py::init<>())
        .def(py::init<const VVT<CT<MT>> &>())
        .def("PrintInfo", &Dim2Matrix<MT>::PrintInfo);
    // basic gate
    py::class_<mindquantum::BasicGate<MT>, std::shared_ptr<mindquantum::BasicGate<MT>>>(m, "basic_gate_cxx")
        .def(py::init<>())
        .def(py::init<bool, std::string, int64_t, Dim2Matrix<MT>>())
        .def(py::init<std::string, bool, MT, MT, MT>())
        .def(py::init<std::string, bool, MT>())
        .def(py::init<std::string, bool, VT<VVT<CT<MT>>>>())
        .def("PrintInfo", &BasicGate<MT>::PrintInfo)
        .def("apply_value", &BasicGate<MT>::ApplyValue)
        .def_readwrite("obj_qubits", &BasicGate<MT>::obj_qubits_)
        .def_readwrite("ctrl_qubits", &BasicGate<MT>::ctrl_qubits_)
        .def_readwrite("params", &BasicGate<MT>::params_)
        .def_readwrite("daggered", &BasicGate<MT>::daggered_)
        .def_readwrite("applied_value", &BasicGate<MT>::applied_value_)
        .def_readwrite("is_measure", &BasicGate<MT>::is_measure_)
        .def_readwrite("base_matrix", &BasicGate<MT>::base_matrix_)
        .def_readwrite("hermitian_prop", &BasicGate<MT>::hermitian_prop_)
        .def_readwrite("is_channel", &BasicGate<MT>::is_channel_)
        .def_readwrite("gate_list", &BasicGate<MT>::gate_list_)
        .def_readwrite("probs", &BasicGate<MT>::probs_)
        .def_readwrite("cumulative_probs", &BasicGate<MT>::cumulative_probs_)
        .def_readwrite("kraus_operator_set", &BasicGate<MT>::kraus_operator_set_);
    m.def("get_gate_by_name", &GetGateByName<MT>);
    m.def("get_measure_gate", &GetMeasureGate<MT>);

    py::class_<BasicGate<MT>, mindquantum::BasicGate<MT>, std::shared_ptr<BasicGate<MT>>>(m, "basic_gate")
        .def(py::init<>())
        .def(py::init<bool, std::string, int64_t, Dim2Matrix<MT>>())
        .def(py::init<std::string, bool, MT, MT, MT>())
        .def(py::init<std::string, bool, MT>())
        .def(py::init<std::string, bool, VT<VVT<CT<MT>>>>())
        .def(py::init<std::string, int64_t, py::object, py::object>())
        .def("PrintInfo", &BasicGate<MT>::PrintInfo)
        .def("apply_value", &BasicGate<MT>::ApplyValue)
        .def_readwrite("obj_qubits", &BasicGate<MT>::obj_qubits_)
        .def_readwrite("ctrl_qubits", &BasicGate<MT>::ctrl_qubits_)
        .def_readwrite("params", &BasicGate<MT>::params_)
        .def_readwrite("daggered", &BasicGate<MT>::daggered_)
        .def_readwrite("applied_value", &BasicGate<MT>::applied_value_)
        .def_readwrite("is_measure", &BasicGate<MT>::is_measure_)
        .def_readwrite("base_matrix", &BasicGate<MT>::base_matrix_)
        .def_readwrite("hermitian_prop", &BasicGate<MT>::hermitian_prop_)
        .def_readwrite("is_channel", &BasicGate<MT>::is_channel_)
        .def_readwrite("gate_list", &BasicGate<MT>::gate_list_)
        .def_readwrite("probs", &BasicGate<MT>::probs_)
        .def_readwrite("cumulative_probs", &BasicGate<MT>::cumulative_probs_)
        .def_readwrite("kraus_operator_set", &BasicGate<MT>::kraus_operator_set_);
    // parameter resolver

    auto real_pr = BindPR<MT>(m, "real_pr");
    auto complex_pr = BindPR<std::complex<MT>>(m, "complex_pr");

    namespace op = bindops::details;

    using real_pr_t = decltype(real_pr);
    using pr_t = real_pr_t::type;
    using complex_pr_t = decltype(complex_pr);
    using pr_cmplx_t = complex_pr_t::type;

    using all_scalar_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t>;

    complex_pr.def("update", &pr_cmplx_t::Update<MT>);

    bindops::binop_definition<op::plus, real_pr_t>::inplace<double, pr_t>(real_pr);
    bindops::binop_definition<op::plus, real_pr_t>::external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::plus, real_pr_t>::reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::inplace<double, pr_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::minus, real_pr_t>::reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::inplace<double, pr_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::times, real_pr_t>::reverse<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::inplace<double, pr_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::external<all_scalar_types_t>(real_pr);
    bindops::binop_definition<op::divides, real_pr_t>::reverse<all_scalar_types_t>(real_pr);

    bindops::binop_definition<op::plus, complex_pr_t>::inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::plus, complex_pr_t>::external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::plus, complex_pr_t>::reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::minus, complex_pr_t>::reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::times, complex_pr_t>::reverse<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::inplace<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::external<all_scalar_types_t>(complex_pr);
    bindops::binop_definition<op::divides, complex_pr_t>::reverse<all_scalar_types_t>(complex_pr);

    // pauli mat
    py::class_<PauliMat<MT>, std::shared_ptr<PauliMat<MT>>>(m, "pauli_mat")
        .def(py::init<>())
        .def(py::init<const PauliTerm<MT> &, Index>())
        .def_readonly("n_qubits", &PauliMat<MT>::n_qubits_)
        .def_readonly("dim", &PauliMat<MT>::dim_)
        .def_readwrite("coeff", &PauliMat<MT>::p_)
        .def("PrintInfo", &PauliMat<MT>::PrintInfo);

    m.def("get_pauli_mat", &GetPauliMat<MT>);

    // csr_hd_matrix
    py::class_<CsrHdMatrix<MT>, std::shared_ptr<CsrHdMatrix<MT>>>(m, "csr_hd_matrix")
        .def(py::init<>())
        .def(py::init<Index, Index, py::array_t<Index>, py::array_t<Index>, py::array_t<CT<MT>>>())
        .def("PrintInfo", &CsrHdMatrix<MT>::PrintInfo);
    m.def("csr_plus_csr", &Csr_Plus_Csr<MT>);
    m.def("transpose_csr_hd_matrix", &TransposeCsrHdMatrix<MT>);
    m.def("pauli_mat_to_csr_hd_matrix", &PauliMatToCsrHdMatrix<MT>);

    // hamiltonian
    py::class_<Hamiltonian<MT>, std::shared_ptr<Hamiltonian<MT>>>(m, "hamiltonian")
        .def(py::init<>())
        .def(py::init<const VT<PauliTerm<MT>> &>())
        .def(py::init<const VT<PauliTerm<MT>> &, Index>())
        .def(py::init<std::shared_ptr<CsrHdMatrix<MT>>, Index>())
        .def_readwrite("how_to", &Hamiltonian<MT>::how_to_)
        .def_readwrite("n_qubits", &Hamiltonian<MT>::n_qubits_)
        .def_readwrite("ham", &Hamiltonian<MT>::ham_)
        .def_readwrite("ham_sparse_main", &Hamiltonian<MT>::ham_sparse_main_)
        .def_readwrite("ham_sparse_second", &Hamiltonian<MT>::ham_sparse_second_);
    m.def("sparse_hamiltonian", &SparseHamiltonian<MT>);

#ifdef ENABLE_PROJECTQ
    using mindquantum::projectq::InnerProduct;
    using mindquantum::projectq::Projectq;

    // projectq simulator
    py::class_<Projectq<MT>, std::shared_ptr<Projectq<MT>>>(m, "projectq")
        .def(py::init<>())
        .def(py::init<unsigned, unsigned>())
        .def("reset", py::overload_cast<>(&Projectq<MT>::InitializeSimulator))
        .def("apply_measure", &Projectq<MT>::ApplyMeasure)
        .def("apply_gate", py::overload_cast<const mindquantum::BasicGate<MT> &>(&Projectq<MT>::ApplyGate))
        .def("apply_gate", py::overload_cast<const mindquantum::BasicGate<MT> &, const ParameterResolver<MT> &, bool>(
                               &Projectq<MT>::ApplyGate))
        .def("apply_circuit", py::overload_cast<const VT<mindquantum::BasicGate<MT>> &>(&Projectq<MT>::ApplyCircuit))
        .def("apply_circuit", py::overload_cast<const VT<mindquantum::BasicGate<MT>> &, const ParameterResolver<MT> &>(
                                  &Projectq<MT>::ApplyCircuit))
        .def("apply_circuit_with_measure", &Projectq<MT>::ApplyCircuitWithMeasure)
        .def("sampling", &Projectq<MT>::Sampling)
        .def("apply_hamiltonian", &Projectq<MT>::ApplyHamiltonian)
        .def("get_expectation", &Projectq<MT>::GetExpectation)
        .def("PrintInfo", &Projectq<MT>::PrintInfo)
        .def("run", &Projectq<MT>::run)
        .def("get_qs", &Projectq<MT>::cheat)
        .def("set_qs", &Projectq<MT>::SetState)
        .def("get_circuit_matrix", &Projectq<MT>::GetCircuitMatrix)
        .def("set_threads_number", &Projectq<MT>::SetThreadsNumber)
        .def("copy", &Projectq<MT>::Copy)
        .def("hermitian_measure_with_grad",
             py::overload_cast<const VT<Hamiltonian<MT>> &, const VT<mindquantum::BasicGate<MT>> &,
                               const VT<mindquantum::BasicGate<MT>> &, const VVT<MT> &, const VT<MT> &, const VS &,
                               const VS &, size_t, size_t>(&Projectq<MT>::HermitianMeasureWithGrad))
        .def("non_hermitian_measure_with_grad",
             py::overload_cast<const VT<Hamiltonian<MT>> &, const VT<Hamiltonian<MT>> &,
                               const VT<mindquantum::BasicGate<MT>> &, const VT<mindquantum::BasicGate<MT>> &,
                               const VT<mindquantum::BasicGate<MT>> &, const VT<mindquantum::BasicGate<MT>> &,
                               const VVT<MT> &, const VT<MT> &, const VS &, const VS &, size_t, size_t,
                               const Projectq<MT> &>(&Projectq<MT>::NonHermitianMeasureWithGrad));
    m.def("cpu_projectq_inner_product", &InnerProduct<MT>);
#endif
}
