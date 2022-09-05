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
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

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

#include "python/PRPython.hpp"

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

void BindPRPython(py::module *m) {
    using mindquantum::MST;
    using mindquantum::PRPython;
    using mindquantum::PRTypeID;

    py::class_<PRPython, std::shared_ptr<PRPython>>(*m, "pr_python")
        .def(py::init<double>())
        .def(py::init<std::complex<double>>())
        .def(py::init<const MST<double>, double>())
        .def(py::init<const MST<std::complex<double>>, std::complex<double>>())
        .def(py::init([](const PRPython &pr) { return std::make_shared<PRPython>(pr); }))
        .def(py::init<std::string, PRTypeID>())
        .def("set_const", py::overload_cast<double>(&PRPython::SetConst))
        .def("set_const", py::overload_cast<std::complex<double>>(&PRPython::SetConst))
        .def("params_name", &PRPython::ParamsName)
        .def("display", &PRPython::PrintInfo)
        .def("__len__", &PRPython::Size)
        .def("size", &PRPython::Size)
        .def("__bool__", &PRPython::IsNotZero)
        .def("__repr__", &PRPython::ToString)
        .def("__str__", &PRPython::ToString)
        .def("__contains__", &PRPython::Contains)
        .def("__copy__", [](const PRPython &pr) { return pr; })
        .def("get_key", &PRPython::GetKey)
        .def(py::self += double(), py::is_operator())
        .def(py::self += std::complex<double>(), py::is_operator())
        .def(py::self += py::self, py::is_operator())
        .def(py::self -= double(), py::is_operator())
        .def(py::self -= std::complex<double>(), py::is_operator())
        .def(py::self -= py::self, py::is_operator())
        .def(py::self *= double(), py::is_operator())
        .def(py::self *= std::complex<double>(), py::is_operator())
        .def(py::self *= py::self, py::is_operator())
        .def(py::self /= double(), py::is_operator())
        .def(py::self /= std::complex<double>(), py::is_operator())
        .def(py::self /= py::self, py::is_operator())
        .def("is_const", &PRPython::IsConst)
        .def("is_const", &PRPython::IsConst)
        .def("requires_grad", &PRPython::RequiresGrad)
        .def("no_grad", &PRPython::NoGrad)
        .def("no_grad_part", &PRPython::NoGradPart)
        .def("requires_grad_part", &PRPython::RequiresGradPart)
        .def("as_encoder", &PRPython::AsEncoder)
        .def("as_ansatz", &PRPython::AsAnsatz)
        .def("encoder_part", &PRPython::EncoderPart)
        .def("ansatz_part", &PRPython::AnsatzPart)
        .def("update", py::overload_cast<const PRPython &>(&PRPython::Update))
        .def("conjugate", &PRPython::Conjugate)
        .def("combination", &PRPython::Combination)
        .def("real", &PRPython::Real)
        .def("imag", &PRPython::Imag)
        // .def("pop", &PRPython::Pop)
        .def("is_hermitian", &PRPython::IsHermitian)
        .def("is_anti_hermitian", &PRPython::IsAntiHermitian)
        .def("to_complex", &PRPython::ToComplexPR)
        .def("is_complex_pr", &PRPython::IsComplexPR);
}

template <typename T>
void BindPR(py::module *m, const std::string &name) {
    using mindquantum::MST;
    using mindquantum::ParameterResolver;
    using mindquantum::SS;

    py::class_<ParameterResolver<T>, std::shared_ptr<ParameterResolver<T>>>(*m, name.c_str())
        .def(py::init<T>())
        .def(py::init([](ParameterResolver<T> &pr, bool copy) {
            if (copy) {
                return pr;
            }
            return std::move(pr);
        }))
        .def(py::init<std::string>())
        .def(py::init<const MST<T> &, T>())
        .def(py::init<const MST<T> &, T, const SS &, const SS &>())
        .def_readonly("const", &ParameterResolver<T>::const_value)
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
        .def("__repr__", &ParameterResolver<T>::ToString)
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

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
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

    m.doc() = "MindQuantum c plugin";

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
    BindPR<MT>(&m, "real_pr");
    BindPR<std::complex<MT>>(&m, "complex_pr");
    BindPRPython(&m);

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
