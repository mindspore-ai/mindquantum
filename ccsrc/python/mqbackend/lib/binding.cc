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
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config/constexpr_type_name.hpp"

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

#include "python/details/create_from_container_class.hpp"

namespace py = pybind11;

using mindquantum::sparse::Csr_Plus_Csr;
using mindquantum::sparse::GetPauliMat;
using mindquantum::sparse::PauliMat;
using mindquantum::sparse::PauliMatToCsrHdMatrix;
using mindquantum::sparse::SparseHamiltonian;
using mindquantum::sparse::TransposeCsrHdMatrix;

#ifdef ENABLE_PROJECTQ
using mindquantum::projectq::InnerProduct;
using mindquantum::projectq::Projectq;
#endif

template <typename T>
auto BindPR(py::module &module, const std::string &name) {
    using mindquantum::MST;
    using mindquantum::ParameterResolver;
    using mindquantum::SS;
    using mindquantum::python::create_from_python_container_class;

    using pr_t = mindquantum::ParameterResolver<T>;

    using namespace pybind11::literals;

    return py::class_<pr_t, std::shared_ptr<pr_t>>(module, name.c_str())
        .def(py::init<T>())
        .def(py::init([](ParameterResolver<T> &pr, bool copy) {
            if (copy) {
                return pr;
            }
            return std::move(pr);
        }))
        .def(py::init<std::string>())
        .def(py::init<const MST<T> &>(), "data"_a)
        .def(py::init<const MST<T> &, T>(), "data"_a, "coeff"_a)
        .def(py::init<const MST<T> &, T, const SS &, const SS &>())
        //! *VERY* important: this overload below needs to be the LAST
        .def(pybind11::init(&create_from_python_container_class<pr_t>), "py_class"_a,
             "Constructor from the encapsulating Python class (using a _cpp_obj attribute)")
        .def_property_readonly("const", [](const ParameterResolver<T> &pr) { return pr.const_value; })
        .def_readonly("data", &ParameterResolver<T>::data_)
        .def_readonly("no_grad_parameters", &ParameterResolver<T>::no_grad_parameters_)
        .def_readonly("encoder_parameters", &ParameterResolver<T>::encoder_parameters_)
        .def("set_const", &ParameterResolver<T>::SetConst)
        .def("params_name", &ParameterResolver<T>::ParamsName)
        .def("display", &ParameterResolver<T>::PrintInfo)
        .def("__setitem__", &ParameterResolver<T>::SetItem)
        .def("__getitem__", py::overload_cast<size_t>(&ParameterResolver<T>::GetItem, py::const_))
        .def("__getitem__", py::overload_cast<const std::string &>(&ParameterResolver<T>::GetItem, py::const_))
        .def("__len__", &ParameterResolver<T>::Size)
        .def("size", &ParameterResolver<T>::Size)
        .def("__bool__", &ParameterResolver<T>::IsNotZero)
        .def("__repr__",
             [](const ParameterResolver<T> &pr) {
                 return fmt::format("ParameterResolver<{}>({})", mindquantum::get_type_name<T>(), pr.ToString());
             })
        .def("__str__", &ParameterResolver<T>::ToString)
        .def("__contains__", &ParameterResolver<T>::Contains)
        .def("__copy__", &ParameterResolver<T>::Copy)
        .def("get_key", &ParameterResolver<T>::GetKey)
        .def(py::self + py::self)
        .def(T() + py::self)
        .def(py::self + T())
        .def(py::self - py::self)
        .def(T() - py::self)
        .def(py::self - T())
        .def(py::self * py::self)
        .def(py::self * T())
        .def(T() * py::self)
        .def(py::self / py::self)
        .def(T() / py::self)
        .def(py::self / T())
        .def(py::self == T())
        .def(py::self == py::self)
        .def(-py::self)
        .def("is_const", &ParameterResolver<T>::IsConst)
        .def("requires_grad", &ParameterResolver<T>::RequiresGrad)
        .def("no_grad", &ParameterResolver<T>::NoGrad)
        .def("no_grad_part", &ParameterResolver<T>::NoGradPart)
        .def("requires_grad_part", &ParameterResolver<T>::RequiresGradPart)
        .def("as_encoder", &ParameterResolver<T>::AsEncoder)
        .def("as_ansatz", &ParameterResolver<T>::AsAnsatz)
        .def("encoder_part", &ParameterResolver<T>::EncoderPart)
        .def("ansatz_part", &ParameterResolver<T>::AnsatzPart)
        .def("update", &ParameterResolver<T>::Update)
        .def("conjugate", &ParameterResolver<T>::Conjugate)
        .def("combination", &ParameterResolver<T>::Combination)
        .def("real", &ParameterResolver<T>::Real)
        .def("imag", &ParameterResolver<T>::Imag)
        .def("pop", &ParameterResolver<T>::Pop)
        .def("is_hermitian", &ParameterResolver<T>::IsHermitian)
        .def("is_anti_hermitian", &ParameterResolver<T>::IsAntiHermitian)
        .def("to_complex", &ParameterResolver<T>::ToComplexPR)
        .def("is_complex_pr", &ParameterResolver<T>::IsComplexPR);
}

namespace mindquantum::python {
void init_logging(pybind11::module &module);
}  // namespace mindquantum::python

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
    using namespace pybind11::literals;
    using mindquantum::BasicGate;
    using mindquantum::CsrHdMatrix;
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

    m.doc() = "MindQuantum C++ plugin";

    py::module logging = m.def_submodule("logging", "MindQuantum-C++ logging module");
    mindquantum::python::init_logging(logging);

    // matrix
    py::class_<Dim2Matrix<MT>, std::shared_ptr<Dim2Matrix<MT>>>(m, "dim2matrix")
        .def(py::init<>())
        .def(py::init<const VVT<CT<MT>> &>())
        .def("PrintInfo", &Dim2Matrix<MT>::PrintInfo);
    // basic gate
    py::class_<BasicGate<MT>, std::shared_ptr<BasicGate<MT>>>(m, "basic_gate")
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
    m.def("get_gate_by_name", &GetGateByName<MT>);
    m.def("get_measure_gate", &GetMeasureGate<MT>);
    // parameter resolver
    auto real_pr = BindPR<MT>(m, "real_pr");
    auto complex_pr = BindPR<std::complex<MT>>(m, "complex_pr");
    complex_pr.def(py::init<MT>())
        .def(py::init<const mindquantum::MST<MT> &>(), "data"_a)
        .def(py::init<const mindquantum::MST<MT> &, MT>(), "data"_a, "coeff"_a = 0.);

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
    // projectq simulator
    py::class_<Projectq<MT>, std::shared_ptr<Projectq<MT>>>(m, "projectq")
        .def(py::init<>())
        .def(py::init<unsigned, unsigned>())
        .def("reset", py::overload_cast<>(&Projectq<MT>::InitializeSimulator))
        .def("apply_measure", &Projectq<MT>::ApplyMeasure)
        .def("apply_gate", py::overload_cast<const BasicGate<MT> &>(&Projectq<MT>::ApplyGate))
        .def("apply_gate",
             py::overload_cast<const BasicGate<MT> &, const ParameterResolver<MT> &, bool>(&Projectq<MT>::ApplyGate))
        .def("apply_circuit", py::overload_cast<const VT<BasicGate<MT>> &>(&Projectq<MT>::ApplyCircuit))
        .def("apply_circuit",
             py::overload_cast<const VT<BasicGate<MT>> &, const ParameterResolver<MT> &>(&Projectq<MT>::ApplyCircuit))
        .def("apply_circuit_with_measure", &Projectq<MT>::ApplyCircuitWithMeasure)
        .def("sampling", &Projectq<MT>::Sampling)
        .def("apply_hamiltonian", &Projectq<MT>::ApplyHamiltonian)
        .def("get_expectation", &Projectq<MT>::GetExpectation)
        .def("PrintInfo", &Projectq<MT>::PrintInfo)
        .def("run", &Projectq<MT>::run)
        .def("get_qs", &Projectq<MT>::cheat)
        .def("set_qs", &Projectq<MT>::SetState)
        .def("get_circuit_matrix", &Projectq<MT>::GetCircuitMatrix)
        .def("copy", &Projectq<MT>::Copy)
        .def("hermitian_measure_with_grad",
             py::overload_cast<const VT<Hamiltonian<MT>> &, const VT<BasicGate<MT>> &, const VT<BasicGate<MT>> &,
                               const VVT<MT> &, const VT<MT> &, const VS &, const VS &, size_t, size_t>(
                 &Projectq<MT>::HermitianMeasureWithGrad))
        .def("non_hermitian_measure_with_grad",
             py::overload_cast<const VT<Hamiltonian<MT>> &, const VT<Hamiltonian<MT>> &, const VT<BasicGate<MT>> &,
                               const VT<BasicGate<MT>> &, const VT<BasicGate<MT>> &, const VT<BasicGate<MT>> &,
                               const VVT<MT> &, const VT<MT> &, const VS &, const VS &, size_t, size_t,
                               const Projectq<MT> &>(&Projectq<MT>::NonHermitianMeasureWithGrad));
    m.def("cpu_projectq_inner_product", &InnerProduct<MT>);
#endif
}
