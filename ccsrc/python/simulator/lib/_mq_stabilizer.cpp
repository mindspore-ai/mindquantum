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
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "simulator/stabilizer/random_benchmarking.h"
#include "simulator/stabilizer/stabilizer.h"

PYBIND11_MODULE(_mq_stabilizer, module) {
    using namespace pybind11::literals;  // NOLINT

    module.doc() = "MindQuantum c++ stabilizer simulator.";
    using namespace mindquantum::stabilizer;  // NOLINT

    pybind11::class_<StabilizerTableau>(module, "StabilizerTableau")
        .def(pybind11::init<size_t, unsigned>(), "n_qubits"_a, "seed"_a = 42)
        .def("copy", [](const StabilizerTableau& s) { return s; })
        .def("tableau_to_string", &StabilizerTableau::TableauToString)
        .def("stabilizer_to_string", &StabilizerTableau::StabilizerToString)
        .def("apply_circuit", &StabilizerTableau::ApplyCircuit)
        .def("decompose", &StabilizerTableau::Decompose)
        .def("apply_gate", &StabilizerTableau::ApplyGate)
        .def("tableau_to_vector", &StabilizerTableau::TableauToVector)
        .def("reset", &StabilizerTableau::Reset)
        .def("sampling", &StabilizerTableau::Sampling)
        .def("sampling_measure_ending_without_noise", &StabilizerTableau::SamplingMeasurementEndingWithoutNoise)
        .def("__eq__", [](const StabilizerTableau& lhs, const StabilizerTableau& rhs) { return lhs == rhs; })
        .def("get_expectation", &StabilizerTableau::GetExpectation, "ham_termlist"_a, "circuit"_a);

    module.def("query_single_qubit_clifford_elem", &QuerySingleQubitCliffordElem, "idx"_a);
    module.def("query_double_qubits_clifford_elem", &QueryDoubleQubitsCliffordElem, "idx"_a);
    module.def("generate_single_qubit_rb_circ", &SingleQubitRBCircuit, "len"_a, "seed"_a);
    module.def("generate_double_qubits_rb_circ", &DoubleQubitsRBCircuit, "len"_a, "seed"_a);
    module.def("verify", &Verification);
}