/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __CUDACC__
#    error "CUDA backend for MQChem not supported yet"
#else
#    include "simulator/chemistry/detail/cpu_ci_vector_double_policy.h"
#    include "simulator/chemistry/detail/cpu_ci_vector_float_policy.h"
#endif

#include "simulator/chemistry/detail/chem_timing.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"

#include "python/chemistry/bind_ci_state.h"
using mindquantum::qbit_t;

namespace py = pybind11;

PYBIND11_MODULE(_mq_chem, module) {
    using namespace py::literals;  // NOLINT
    std::string sim_name = "mqchem";

#ifndef __CUDACC__
    using float_policy_t = mindquantum::sim::chem::detail::cpu_ci_vector_float_policy;
    using double_policy_t = mindquantum::sim::chem::detail::cpu_ci_vector_double_policy;
#endif

    using float_chem_sim = mindquantum::sim::chem::CIState<float_policy_t>;
    using double_chem_sim = mindquantum::sim::chem::CIState<double_policy_t>;

    module.doc() = "MindQuantum C++ Chemistry simulator.";
    py::module float_sim = module.def_submodule("float", "float precision MQChem simulator");
    py::module double_sim = module.def_submodule("double", "double precision MQChem simulator");

    BindSim<float_chem_sim>(float_sim, sim_name).def("sim_name", [sim_name](const float_chem_sim&) {
        return sim_name;
    });
    BindSim<double_chem_sim>(double_sim, sim_name).def("sim_name", [sim_name](const double_chem_sim&) {
        return sim_name;
    });

    // Bind CppExcitationOperator for float and double precision
    using float_cpp_exc_t = mindquantum::sim::chem::detail::CppExcitationOperator<float>;
    py::class_<float_cpp_exc_t, std::shared_ptr<float_cpp_exc_t>>(float_sim, "CppExcitationOperator")
        .def(py::init<const float_cpp_exc_t::FermionOpData&, qbit_t, int, const parameter::ParameterResolver&>(),
             "term_data"_a, "n_qubits"_a, "n_electrons"_a, "pr"_a = parameter::ParameterResolver());

    using double_cpp_exc_t = mindquantum::sim::chem::detail::CppExcitationOperator<double>;
    py::class_<double_cpp_exc_t, std::shared_ptr<double_cpp_exc_t>>(double_sim, "CppExcitationOperator")
        .def(py::init<const double_cpp_exc_t::FermionOpData&, qbit_t, int, const parameter::ParameterResolver&>(),
             "term_data"_a, "n_qubits"_a, "n_electrons"_a, "pr"_a = parameter::ParameterResolver());

    // Bind CppCIHamiltonian for float and double precision
    using float_cpp_ci_t = mindquantum::sim::chem::detail::CppCIHamiltonian<float>;
    py::class_<float_cpp_ci_t>(float_sim, "CppCIHamiltonian")
        .def(py::init<const float_cpp_ci_t::FermionOpData&, qbit_t, int>(), "ham_data"_a, "n_qubits"_a,
             "n_electrons"_a);
    using double_cpp_ci_t = mindquantum::sim::chem::detail::CppCIHamiltonian<double>;
    py::class_<double_cpp_ci_t>(double_sim, "CppCIHamiltonian")
        .def(py::init<const double_cpp_ci_t::FermionOpData&, qbit_t, int>(), "ham_data"_a, "n_qubits"_a,
             "n_electrons"_a);

    // Expose timing functions
    module.def(
        "print_timing_report", []() { mindquantum::sim::chem::detail::ChemTimer::getInstance().printReport(); },
        "Print the cumulative timing report for ApplyUCCGate stages");

    module.def(
        "reset_timing", []() { mindquantum::sim::chem::detail::ChemTimer::getInstance().reset(); },
        "Reset all timing data");
}
